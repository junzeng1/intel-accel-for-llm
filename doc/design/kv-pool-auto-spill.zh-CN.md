# KV Cache Memory Pool 水位限制与后台自动持久化/淘汰设计文档

## 1. 背景与设计目标

在依托 vLLM 与 `iaxl`（`KVShrinkConnector` + `KVStore`）的 KV Cache 以存代算方案中，涵盖了两种核心部署形态（对应 G3.5 两套参考方案）：

1. **G3.5 方案一（本地 HeadNode CPU 缓存 + 远端 JBOF/NVMe-oF 存储）**：
   - KV Cache 经由 HeadNode 本地的 CUDA / Intel DSA 接口导出到 HeadNode DDR 内存池（`KVStoreLocal`），由本地 QAT / IAA 压缩后暂存，并可进一步落盘至挂载在本地的远端 JBOF（SPDK NVMe-oF）目录。
2. **G3.5 方案二（跨节点 GPU Direct RDMA + 远端存储节点本地 NVMe）**：
   - 旁路 HeadNode CPU，vLLM Worker 通过 `KVStoreRemote` 经由 GDR / RDMA 将 KV Cache 直接传输至远端 Daemon 存储节点，在 Daemon 节点的 각 Rank 进程内复用完全相同的 `KVStoreLocal` -> `KVFlow` -> `kv_pool::Mem` 栈进行 QAT / IAA 压缩、DDR 缓存池管理及本地 NVMe 持久化。

在原有实现中，`kv_pool::Mem` 虽然支持通过 REST/CLI 手动触发 `persist` 与 `evict`，但默认不会限制 DDR 内存池的实际占用上限，也不会在后台自动执行持久化与驱逐。

本特性在 C++ `kv_pool` 层（[`iaxl/csrc/kv_pool/mem.cpp`](../../iaxl/csrc/kv_pool/mem.cpp)、[`storage.cpp`](../../iaxl/csrc/kv_pool/storage.cpp)、[`record.cpp`](../../iaxl/csrc/kv_pool/record.cpp)）设计并实现了**水位驱动的常驻低优先级后台自动 Persist + Evict（以及纯 Evict-Only）机制**，满足以下核心设计目标：

1. **默认关闭，按需开启**：默认保持原有纯内存行为（`IAXL_CACHE_AUTO_SPILL_ENABLE=0`），不启动后台线程；开启后按高低水位滞回（Hysteresis）自动控制。
2. **同时支持两种后台工作模式**：
   - **Persist + Evict 模式（`IAXL_CACHE_AUTO_SPILL_EVICT_ONLY=0`，对应测试 Case C）**：达到高水位（默认 80%）时，后台针对最老的 LRU Group 先异步持久化到后端存储，落盘并更新元数据后再从 DDR 内存池驱逐，直至降至低水位（默认 60%）暂停。后续复访自动从存储加载（`hits_in_storage`）。
   - **Direct Evict-Only 模式（`IAXL_CACHE_AUTO_SPILL_EVICT_ONLY=1`，对应测试 Case B）**：达到高水位时，后台不写后端存储，直接异步淘汰最老的 LRU Group 并同步摘除 `Record` 元数据，直至降至低水位暂停。
3. **对前台 `PUT` / `GET` 零阻塞或极轻微影响**：
   - 锁粒度微步化（Single-Group Micro-Step），消除持锁磁盘 I/O；
   - 后台线程绑定本 Rank 所属 NUMA CPU 核，降低 CPU `nice` 优先级与 Linux I/O 优先级（Best-Effort 最低档），并在检测到前台活跃 `GET` 时自动礼让退避（Pacing）。
4. **本地与跨节点（`remote_pool`）双模式 100% 复用**。
5. **存储后端抽象解耦（`StorageBackend`）**：为后续直接对接支持 KV 接口的 NVMe SSD / SPDK 预留批量存取接口。

---

## 2. 架构与核心流程设计

```mermaid
flowchart TD
    subgraph Foreground["前台数据面 (PUT / GET)"]
        Put["zip_to_mem -> Mem::put"] --> PutLock["持短锁插入/更新 EntryGroup\n(++version, ++access_seq, 移至 LRU 队尾)"]
        PutLock --> CheckWM{"current_bytes >=\nhigh_watermark?"}
        CheckWM -->|Yes| Wake["spill_cv_.notify_one()"]
        Get["unzip_from_mem -> Mem::get"] --> GetP1["Phase 1 [短锁]: 查内存 cache_\n命中则移至 LRU 队尾返回"]
        GetP1 -->|Miss| GetP2["Phase 2 [无锁 & 临时释放 deletion_guard]:\nRecord::is_persisted (read_db_)\n+ StorageBackend::load_batch (OMP 并行)"]
        GetP2 --> GetP3["Phase 3 [恢复 deletion_guard + 短锁]:\nput_loaded_one_unlocked 插回内存"]
    end

    subgraph Background["后台管控面 (Mem::spill_worker_ 'IAXL-Spill')"]
        Wake --> Loop{"current_bytes >\nlow_watermark?"}
        Loop -->|No| Sleep["休眠等待下次高水位唤醒"]
        Loop -->|Yes, EVICT_ONLY=0\n(Case C: Persist+Evict)| ModeC["单 Group 微步:\n1. [短锁] 取最老 LRU Group 快照 (is_persisting=true)\n2. [无锁] Storage::save_batch (.tmp + rename)\n3. [无锁] Record::mark_persisted\n4. [短锁] 校验 version & access_seq 未变则 Evict"]
        Loop -->|Yes, EVICT_ONLY=1\n(Case B: Direct Evict)| ModeB["两阶段 Unpublish-then-Free:\n1. [短锁] 取最老 LRU 批次 (is_unpublishing=true)\n2. [无锁] Record::remove + 等待 grace_ms 窗口\n3. [短锁] 校验 access_seq 未被在途 GET 命中则物理释放"]
        ModeC --> Loop
        ModeB --> Loop
    end
```

---

## 3. 关键并发安全与性能优化细节

### 3.1 消除持锁磁盘 I/O（三阶段无锁化）

在原有实现中，`Mem::get()`、`Mem::persist_groups()` 和 `Mem::evict_groups()` 均在持有 `Mem::mutex_` 的情况下直接调用 `Storage::load()` / `Storage::save()` 及 `Record::is_persisted()`，导致慢速磁盘 I/O 阻塞所有并发的 `PUT` / `GET` 压缩线程。

本方案将三者全部改造为 **Three-Phase（短锁元数据快照 -> 无锁磁盘与 SQLite I/O -> 短锁状态提交）**：

- **`Mem::get()` 与精细化 `deletion_guard`**：
  - 当 `UnzipFromMemWork` 调用 `mem->get(keys, true, /*caller_holds_deletion_guard=*/true)` 时：
  - 若 Phase 1 全部命中 DDR 内存，直接微秒级返回指针；
  - 若存在内存未命中需进入 Phase 2 读盘，在进入无锁读盘前**临时释放调用方持有的 `deletion_guard`**，待磁盘 `load_batch()` 完成后、进入 Phase 3 加 `mutex_` 解析最终裸指针前**重新获取 `deletion_guard`**。
  - **收益**：`deletion_guard` 仅在解压读取内存裸指针的几十微秒内持有，慢速读盘期间绝不阻塞后台线程淘汰其他冷 Group。

### 3.2 版本校验（`version`）与热度保活校验（`access_seq`）

每个 `EntryGroup` 维护两个单调递增序号：
- `uint64_t version`：每次 `put_one_unlocked` 写入或覆盖该 Group 的层数据时递增；
- `uint64_t access_seq`：每次 `put_one_unlocked` 或 `get_one_unlocked` 命中该 Group 时赋予全局递增序号 `++global_access_seq_`，同时将该 Group 移至 `lru_head_` 队尾（MRU）。

在后台线程对最老的 `Group_A` 执行 **Persist -> Evict** 时：
1. **Persist 开始前（Step 1）**：记录 `snap_version = Group_A.version` 与 `snap_access_seq = Group_A.access_seq`。
2. **Persist 完成后、准备 Evict 前（Step 3）**：
   - **若期间有新的 `PUT` 更新了该 Group（`Group_A.version != snap_version`）**：不标记 `is_persisted = true`，也不执行 Evict，留待后续轮次重新持久化。
   - **若期间有并发 `GET` 读取了该 Group（`Group_A.access_seq != snap_access_seq`）**：
     - 因为数据内容未变，保留 `Group_A.is_persisted = true` 持久化成果；
     - 但因为该 Group 刚被读取变热且已被 `get_one_unlocked` 移到了 LRU 队尾，**后台线程立即跳过对该 Group 的 Evict**，使其继续留在 DDR 内存中服务后续请求，短时间内绝不会被再次淘汰。

### 3.3 `EVICT_ONLY=1`（Case B）下的两阶段 Unpublish-then-Free 防竞态机制

在 vLLM 架构中，Scheduler 进程先通过 `Record::has()` 查询 SQLite 判断块是否存在，随后调度 Worker 进程执行 `get()` -> `unzip_from_mem()`。
若在 `EVICT_ONLY=1` 模式下直接同时删除 `Mem` 和 `Record`，处于 Scheduler `has()`（返回 `True`）与 Worker `get()` 之间的在途请求会因既不在内存、又未持久化到磁盘而触发 `unzip_from_mem: cache key missing after retry` 断言失败。

为此，`spill_step_evict_only()` 采用两阶段流水线：
1. **Stage 1（Unpublish）**：持锁选中最老的未持久化 LRU Groups，标记 `is_unpublishing = true`（此时数据**仍留在 `Mem` 内存中**），释放锁调用 `record_->remove()` 从 SQLite 删除记录。此后 Scheduler 的 `has()` 立即返回 `False`，不再产生新调度。
2. **Stage 2（Grace Window + Physical Free）**：等待 `IAXL_CACHE_EVICT_GRACE_MS`（默认 50ms）排空窗口，让已通过 `has()` 的在途 Worker `GET` 顺利在内存中命中；随后持锁检查 `access_seq`，仅对窗口期内未被在途 `GET` 访问的 Group 执行物理释放；若被在途 `GET` 命中则保留在内存并重新向 `Record` 注册。

### 3.4 各 Worker 进程后台线程的 CPU 绑核与低优先级隔离

- 每个 Worker Rank 进程拥有独立的 `Mem` 实例和独立的单后台线程 `"IAXL-Spill"`。
- 线程启动时调用 `TaskQueue::bind_cpus()` 绑定到本 Rank 专属的 `IAXL_CPU_OMP_THREADS_BIND` 核集合，确保与该 Rank 的 GPU、DDR 内存池、QAT/DSA 处于同一 NUMA 节点，无跨 Socket 访问。
- 后台线程不使用 OpenMP 并行（避免与 `omp_queue` 压缩线程争抢线程团队），并通过 `setpriority(PRIO_PROCESS, 0, 10)`（`nice +10`）和 `SYS_ioprio_set`（Best-Effort 最低级 7）降权，自然运行在每个 Rank 由 `IAXL_RESERVED_CPU_NUM` 预留的 CPU 余量上。

### 3.5 `Record` (SQLite) 读写分离与冲突保护

- **Upsert 保序**：`Record::submit()` 使用 `INSERT INTO chunks ... ON CONFLICT(chunk_label) DO UPDATE SET created_at = excluded.created_at`，防止异步 `submit()` 覆盖已持久化的 `persisted = 1` 状态；`mark_persisted()` 使用 `ON CONFLICT DO UPDATE SET persisted = 1`。
- **独立只读连接 `read_db_`**：在 WAL 模式下为 `is_persisted()` 和 `has()` 维护独立的 `read_db_` 连接，使 `Mem::get()` Phase 2 查询持久化状态时无需阻塞等待 `submit()` 写队列。

---

## 4. 存储后端抽象（`StorageBackend`）

在 [`iaxl/csrc/include/kv_pool.h`](../../iaxl/csrc/include/kv_pool.h) 中定义了面向 Key（`full_label`）的抽象基类 `kv_pool::StorageBackend`：

```cpp
class StorageBackend {
  public:
    struct WriteItem {
        std::string full_label;
        const char *data;
        size_t size;
    };

    virtual ~StorageBackend() = default;
    virtual bool save(const std::string &full_label, const char *data, size_t size) = 0;
    virtual std::pair<char *, size_t> load(const std::string &full_label) = 0;
    virtual bool remove(const std::string &full_label);
    virtual std::vector<bool> save_batch(const std::vector<WriteItem> &items);
    virtual std::vector<std::pair<char *, size_t>>
    load_batch(const std::vector<std::string> &full_labels, bool allow_parallel = false);
};
```

- 默认实现 `FileStorageBackend` 在 `save()` 时先写入同目录临时文件 `{path}.bin.tmp.{pid}_{seq}`，校验写入长度无误后通过 `std::filesystem::rename` 原子替换目标文件，同时完整保留原有的 `std::abort()` 强校验以便第一时间暴露元数据异常。
- 后续对接支持 KV 接口的 NVMe SSD 或 SPDK 时，仅需新增继承自 `StorageBackend` 的子类并注入 `Storage`，上层 `Mem`、`KVFlow`、`KVStoreLocal`、`remote_pool` 均无需改动。
