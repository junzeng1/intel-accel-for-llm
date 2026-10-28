# KV Cache 内存池容量限制与后台自动持久化/淘汰 —— 使用指南

本文档介绍如何配置和使用 `iaxl` KVStore 内存池的容量限制与后台高低水位自动持久化/淘汰（Auto-Spill / Auto-Evict）功能，分别涵盖：

- **G3.5 方案一（本地模式）**：HeadNode 本地 CPU DDR 缓存池 + 远端 JBOF（NVMe-oF 挂载目录）或本地 NVMe 存储；
- **G3.5 方案二（跨节点 `remote_pool` 模式）**：通过 GPU Direct RDMA（GDR）将 KV Cache 直传至远端 Daemon 存储节点 DDR 缓存池 + 远端节点本地 NVMe 存储；
- **基准测试 Case A / Case B / Case C** 在上述两种场景下的具体配置与执行步骤。

相关架构设计详见 [KV Cache Memory Pool 水位限制与后台自动持久化/淘汰设计文档](../design/kv-pool-auto-spill.zh-CN.md)。

---

## 一、 环境变量说明

所有配置项均在 `setvars.sh` 与 `iaxl/envs.py` 中提供默认值。**该功能默认关闭（`IAXL_CACHE_AUTO_SPILL_ENABLE=0`）**，未开启时完全保持原有纯内存缓存逻辑。

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `IAXL_DDR_POOL_SIZE_GB` | 未设置（默认取当前可用内存的 1/10） | **每个 Worker Rank** 的 DDR 内存池最大容量上限（单位：GB，支持浮点数）。例如 TP=4 且设为 `16` 时，4 个 Rank 共限制使用 64 GB DDR |
| `IAXL_CACHE_DIR` | `_data/kvcache` | 持久化存储根目录。每个 Rank 会在其下自动创建 `compressed/<model>_rank<r>/`（或 `raw/<model>_rank<r>/`）存放 `chunks.db` 与 `chunks/` 数据 |
| `IAXL_CACHE_AUTO_SPILL_ENABLE` | `0` | **后台水位控制总开关**：<br>• `0`（默认）：关闭自动水位控制，不做自动 Persist 或 Evict；<br>• `1`：开启后台水位控制线程 `"IAXL-Spill"` |
| `IAXL_CACHE_AUTO_SPILL_EVICT_ONLY` | `0` | **后台水位触发后的行为模式**（仅当 `IAXL_CACHE_AUTO_SPILL_ENABLE=1` 时生效）：<br>• `0`（默认，**对应 Case C**）：达到高水位时，后台先将最老的 LRU KV Cache 异步 **Persist** 到 `IAXL_CACHE_DIR`，完成后再 **Evict** 释放内存；<br>• `1`（**对应 Case B**）：达到高水位时，后台**不执行 Persist**，直接异步 **Evict** 最老的 LRU KV Cache 并从元数据中摘除 |
| `IAXL_CACHE_HIGH_WATERMARK` | `0.80` | 触发后台异步任务的内存高水位比例（支持 `0.80` 或 `80` 写法，代表 80%） |
| `IAXL_CACHE_LOW_WATERMARK` | `0.60` | 暂停后台异步任务的内存低水位比例（支持 `0.60` 或 `60` 写法，代表 60%） |
| `IAXL_CACHE_SPILL_BATCH_GROUPS` | `16` | 单轮批量处理的 Cache Group 数量上限 |
| `IAXL_CACHE_SPILL_PACE_MS` | `2` | 当检测到前台存在活跃 `GET` 请求时，后台线程每步之间的礼让休眠毫秒数 |
| `IAXL_CACHE_EVICT_GRACE_MS` | `50` | 在 `IAXL_CACHE_AUTO_SPILL_EVICT_ONLY=1`（Case B）模式下，从 SQLite 摘除元数据到物理释放内存之间的安全排空窗口（毫秒） |

> **注意**：在本地模式（G3.5 方案一）下，上述环境变量设置在启动 vLLM 的容器环境中；在跨节点模式（G3.5 方案二 `remote_pool`）下，DDR 池、压缩与持久化全部运行在 **Daemon 存储节点**上，因此上述环境变量必须设置在 **Daemon 节点**启动 `examples/kvshrink-daemon.sh` 的环境中。

---

## 二、 G3.5 方案一：本地 HeadNode CPU 缓存 + JBOF 存储（`IAXL_RDMA_ENABLE=0`）

在方案一中，vLLM 与 `KVStoreLocal` 运行在同一台 GPU HeadNode 上，通过 PCIe / Intel DSA 将 KV Cache 导出至 HeadNode DDR 内存池，由本地 QAT / IAA 压缩，并将远端 JBOF（SPDK NVMe-oF）挂载为本地目录（如 `/mnt/jbof/kvcache`）。

### 2.1 启动容器环境

```bash
./start.sh
```

### 2.2 方案一下执行 Case A / Case B / Case C 的操作步骤

#### （1）执行 Case A（Baseline：原生 vLLM，无外部 KV Cache 复用）

在容器内不挂载 `--kv-transfer-config` 直接启动标准 vLLM 服务：

```bash
source setvars.sh
export MODEL=/path/to/Qwen2.5-32B-Instruct
export TP_SIZE=4

vllm serve "$MODEL" \
    --trust-remote-code \
    -tp "$TP_SIZE" \
    --gpu-memory-utilization 0.8 \
    --max-model-len "${MAX_MODEL_LEN:-32765}" \
    --port "${PORT:-8000}" \
    --no-enable-prefix-caching
```

#### （2）执行 Case B（受限 DDR 容量复用：达到水线后台直接 Evict，不进行 Persist）

在 Case B 中，开启 `KVShrinkConnector` 并限制每个 Rank 的 DDR 内存池大小（例如每 Rank `8` GB，TP=4 共 32 GB），同时设置 `IAXL_CACHE_AUTO_SPILL_ENABLE=1` 和 **`IAXL_CACHE_AUTO_SPILL_EVICT_ONLY=1`**：

```bash
# 清理上一次残留的元数据目录
rm -rf /mnt/jbof/kvcache/*

export MODEL=/path/to/Qwen2.5-32B-Instruct
export TP_SIZE=4
export IAXL_RDMA_ENABLE=0

# 加速器配置（按主机实际硬件开启 QAT / IAA / DSA）
export IAXL_KV_COMPRESSION=1
export IAXL_QAT_ZIP_ENABLE=1
export IAXL_DSA_GD_ENABLE=1

# Case B 核心配置：限制 DDR 容量，开启水位控制，且设为只 Evict 不 Persist
export IAXL_DDR_POOL_SIZE_GB=8
export IAXL_CACHE_DIR=/mnt/jbof/kvcache
export IAXL_CACHE_AUTO_SPILL_ENABLE=1
export IAXL_CACHE_AUTO_SPILL_EVICT_ONLY=1
export IAXL_CACHE_HIGH_WATERMARK=0.80
export IAXL_CACHE_LOW_WATERMARK=0.60

./examples/kvshrink-vllm-serve.sh
```

启动后可通过管理接口确认 `auto_spill.enabled=true` 且 `auto_spill.evict_only=true`：

```bash
curl -s http://localhost:18700/v1/cache/status | jq '.workers["0"].auto_spill'
```

当内存池使用率达到 80% 时，后台线程会自动淘汰最老的 KV Cache Group 直至使用率回到 60%，期间 `auto_persisted_groups` 保持为 `0`，`auto_evicted_groups` 随淘汰递增。

#### （3）执行 Case C（DDR + JBOF 分层复用：达到水线后台异步 Persist + Evict）

在 Case C 中，保持与 Case B 相同的 `IAXL_DDR_POOL_SIZE_GB`，将 **`IAXL_CACHE_AUTO_SPILL_EVICT_ONLY` 设为 `0`**（默认值）：

```bash
# 清理上一次残留的缓存文件与元数据
rm -rf /mnt/jbof/kvcache/*

export MODEL=/path/to/Qwen2.5-32B-Instruct
export TP_SIZE=4
export IAXL_RDMA_ENABLE=0

export IAXL_KV_COMPRESSION=1
export IAXL_QAT_ZIP_ENABLE=1
export IAXL_DSA_GD_ENABLE=1

# Case C 核心配置：限制 DDR 容量，开启水位控制，先 Persist 落盘再 Evict
export IAXL_DDR_POOL_SIZE_GB=8
export IAXL_CACHE_DIR=/mnt/jbof/kvcache
export IAXL_CACHE_AUTO_SPILL_ENABLE=1
export IAXL_CACHE_AUTO_SPILL_EVICT_ONLY=0
export IAXL_CACHE_HIGH_WATERMARK=0.80
export IAXL_CACHE_LOW_WATERMARK=0.60

./examples/kvshrink-vllm-serve.sh
```

当内存使用率达到 80% 时，后台线程会将最老的 LRU Group 写入 `/mnt/jbof/kvcache` 并在落盘完成后将其从 DDR 淘汰至 60% 低水位；当后续请求再次访问这些已被淘汰出 DDR 的前缀时，会自动从 `/mnt/jbof/kvcache` 异步读回并累加 `hits_in_storage` 计数。

---

## 三、 G3.5 方案二：跨节点 GPU Direct RDMA + 远端存储节点（`IAXL_RDMA_ENABLE=1`）

在方案二中，拓扑包含两台节点（详见 [跨节点 KV Cache 复用文档](remote-pool-kvstore.zh-CN.md)）：
- **Daemon 存储节点**（如 `10.10.10.10`，无 GPU，配备 Xeon CPU + QAT/IAA + 本地 NVMe 盘挂载于 `/mnt/nvme/kvcache`）；
- **Client 推理节点**（如 `10.10.10.11`，配备 4 张 GPU 运行 vLLM）。

> **关键原则**：在方案二中，整个 `KVStore`（DDR 内存池、压缩、后台 Spill/Evict 线程、本地 NVMe 读写）均运行在 **Daemon 存储节点**上。因此切换 Case B 与 Case C 的水位和 `EVICT_ONLY` 环境变量均在 **Daemon 节点**配置；运维 REST 接口也监听在 Daemon 节点 `http://10.10.10.10:18700/v1/cache/*`。

### 3.1 方案二下执行 Case A / Case B / Case C 的操作步骤

#### （1）执行 Case A（Baseline：原生 vLLM，无外部 KV Cache 复用）

无需启动 Daemon，直接在 **Client 节点（10.10.10.11）** 容器内启动不带 connector 的标准 vLLM：

```bash
source setvars.sh
export MODEL=/path/to/Qwen2.5-32B-Instruct
export TP_SIZE=4

vllm serve "$MODEL" \
    --trust-remote-code \
    -tp "$TP_SIZE" \
    --gpu-memory-utilization 0.8 \
    --max-model-len "${MAX_MODEL_LEN:-32765}" \
    --port "${PORT:-8000}" \
    --no-enable-prefix-caching
```

#### （2）执行 Case B（跨节点受限 DDR 容量复用：Daemon 侧后台直接 Evict，不进行 Persist）

**第一步：在 Daemon 节点（10.10.10.10）启动 `kvshrink-daemon.sh`**

```bash
# 宿主机启动无 GPU 容器
NVIDIA_RUNTIME=none ./start.sh

# 容器内执行：
rm -rf /mnt/nvme/kvcache/*

export IAXL_RDMA_ENABLE=1
export IAXL_RDMA_DAEMON_NIC_IPS=10.10.10.10
export IAXL_RDMA_DAEMON_PORT=5555
export IAXL_RDMA_TP_SIZE=4
export TP_SIZE=4

# 配置 Daemon 侧 QAT/IAA 压缩
export IAXL_KV_COMPRESSION=1
export IAXL_QAT_ZIP_ENABLE=1
export KVSHRINK_QAT_DEVICES="0|1|4|5"
export IAXL_SCRATCH_POOL_SIZE_GB=16

# Case B 核心配置：限制 Daemon 各 Rank 的 DDR 容量，开启后台直接 Evict（不落盘）
export IAXL_DDR_POOL_SIZE_GB=8
export IAXL_CACHE_DIR=/mnt/nvme/kvcache
export IAXL_CACHE_AUTO_SPILL_ENABLE=1
export IAXL_CACHE_AUTO_SPILL_EVICT_ONLY=1
export IAXL_CACHE_HIGH_WATERMARK=0.80
export IAXL_CACHE_LOW_WATERMARK=0.60

./examples/kvshrink-daemon.sh
```

**第二步：在 Client 节点（10.10.10.11）启动 vLLM**

```bash
./start.sh

# 容器内执行：
export IAXL_RDMA_ENABLE=1
export IAXL_RDMA_DAEMON_NIC_IPS=10.10.10.10
export IAXL_RDMA_DAEMON_PORT=5555
export IAXL_RDMA_CLIENT_NIC_IPS=10.10.10.11
export IAXL_RDMA_TP_SIZE=4
export TP_SIZE=4
export MODEL=/path/to/Qwen2.5-32B-Instruct

./examples/kvshrink-vllm-serve.sh
```

可通过 `curl -s http://10.10.10.10:18700/v1/cache/status | jq '.workers["0"].auto_spill'` 验证 Daemon 侧处于 `enabled: true, evict_only: true` 模式。

#### （3）执行 Case C（跨节点 DDR + 远端本地 NVMe 分层复用：Daemon 侧后台 Persist + Evict）

**第一步：在 Daemon 节点（10.10.10.10）将 `IAXL_CACHE_AUTO_SPILL_EVICT_ONLY` 设为 `0` 并重启 Daemon**

```bash
rm -rf /mnt/nvme/kvcache/*

export IAXL_RDMA_ENABLE=1
export IAXL_RDMA_DAEMON_NIC_IPS=10.10.10.10
export IAXL_RDMA_DAEMON_PORT=5555
export IAXL_RDMA_TP_SIZE=4
export TP_SIZE=4

export IAXL_KV_COMPRESSION=1
export IAXL_QAT_ZIP_ENABLE=1
export KVSHRINK_QAT_DEVICES="0|1|4|5"
export IAXL_SCRATCH_POOL_SIZE_GB=16

# Case C 核心配置：开启后台先 Persist 到本地 NVMe 再 Evict
export IAXL_DDR_POOL_SIZE_GB=8
export IAXL_CACHE_DIR=/mnt/nvme/kvcache
export IAXL_CACHE_AUTO_SPILL_ENABLE=1
export IAXL_CACHE_AUTO_SPILL_EVICT_ONLY=0
export IAXL_CACHE_HIGH_WATERMARK=0.80
export IAXL_CACHE_LOW_WATERMARK=0.60

./examples/kvshrink-daemon.sh
```

**第二步：在 Client 节点（10.10.10.11）启动 vLLM（命令与 Case B 完全相同）**。

---

## 四、 管理接口（REST API）与状态监控

管理 Controller 默认监听在 `18700` 端口（方案一在 `localhost:18700`，方案二在 `daemon_ip:18700`）：

1. **查询各 Rank 水位与后台自动下刷/淘汰状态**：
   ```bash
   curl -s http://<mgmt_host>:18700/v1/cache/status | jq .
   ```
   每个 Worker Rank 返回的 `auto_spill` 字段示例如下：
   ```json
   "auto_spill": {
     "enabled": true,
     "evict_only": false,
     "running": false,
     "high_watermark": 0.8,
     "low_watermark": 0.6,
     "high_watermark_bytes": 6871947673,
     "low_watermark_bytes": 5153960755,
     "batch_groups": 16,
     "pace_ms": 2,
     "evict_grace_ms": 50,
     "spill_rounds": 6,
     "auto_persisted_groups": 96,
     "auto_persisted_bytes": 805306368,
     "auto_evicted_groups": 96,
     "auto_evicted_bytes": 805306368
   }
   ```
2. **手动触发按 LRU 组合下刷/淘汰（`POST /v1/cache/spill`）**：
   - 按当前模式将内存池水位整理至 60% 低水位：
     ```bash
     curl -s -X POST http://<mgmt_host>:18700/v1/cache/spill \
       -H 'Content-Type: application/json' \
       -d '{"target_usage": 0.60}'
     ```
   - 显式指定仅淘汰不持久化（`evict_only: true`）：
     ```bash
     curl -s -X POST http://<mgmt_host>:18700/v1/cache/spill \
       -H 'Content-Type: application/json' \
       -d '{"target_usage": 0.60, "evict_only": true}'
     ```

---

## 五、 使用 `tests/vllm-benchmark-tiering-poc.sh` 运行分层存储基准测试

仓库下提供了专门面向 Case A / B / C 对比测试的脚本 [`tests/vllm-benchmark-tiering-poc.sh`](../../tests/vllm-benchmark-tiering-poc.sh)。该脚本默认采用 `revisit` 三阶段工作流（**Stage 1 目标前缀预热 -> Stage 2 压力前缀冲刷触发后台 Evict 或 Persist+Evict -> Stage 3 复访目标前缀并测量 TTFT/吞吐**）。

### 5.1 针对当前已启动的单组服务执行测试（自动识别 Case A / B / C）

分别按上文配置启动 Case A、Case B 或 Case C 服务后，指定相同的 `LOG_DIR` 依次运行：

```bash
# 方案一（本地模式）：REMOTE_CACHE_HOST 默认为 localhost
# 方案二（remote_pool 模式）：导出 IAXL_RDMA_DAEMON_NIC_IPS=10.10.10.10（或 REMOTE_CACHE_HOST=10.10.10.10）
export MODEL=/path/to/Qwen2.5-32B-Instruct
export LOG_DIR=logs_tiering_poc_run1

# 脚本会自动探测当前运行的服务属于 Case A、B 还是 C，并将结果追加写入 $LOG_DIR/benchmark_summary.csv
bash tests/vllm-benchmark-tiering-poc.sh
```

当 `LOG_DIR` 下积累了 Case A、Case B 和 Case C 的测试记录后，脚本会自动生成：
- `$LOG_DIR/benchmark_summary.csv`：包含各测试点的 Mean/P50/P95/P99 TTFT、TPOT、吞吐、`hits_delta`、`hits_in_storage_delta`、`persisted_delta`、`evicted_delta`、块设备读带宽与压缩/解压吞吐；
- `$LOG_DIR/case_comparison.csv`：逐个（`input_len`, `concurrency`, `hit_rate`）对比 Case C 相对 Case A 和 Case B 的 P95/P99 TTFT 改善幅度；
- `$LOG_DIR/conclusion.md`：自动判定结论报告。

### 5.2 通过启停钩子一键自动跑完 Case A / B / C

如果提供了自动启停脚本（在脚本内读取导出的 `CASE_ID`、`CASE_KVSHRINK_ENABLE`、`IAXL_CACHE_AUTO_SPILL_ENABLE`、`IAXL_CACHE_AUTO_SPILL_EVICT_ONLY`），可一次性自动完成 A/B/C 三组完整矩阵测试：

```bash
CASES="A B C" \
SERVER_LIFECYCLE_MODE=restart \
START_CASE_CMD='./my_start_service.sh' \
STOP_CASE_CMD='./my_stop_service.sh' \
MODEL=/path/to/Qwen2.5-32B-Instruct \
bash tests/vllm-benchmark-tiering-poc.sh
```
