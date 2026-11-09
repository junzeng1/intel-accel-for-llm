// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "kv_pool.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "[FAIL] %s (%s:%d)\n", (msg), __FILE__, __LINE__);                \
            ++g_failures;                                                                          \
        } else {                                                                                   \
            std::printf("[ok]   %s\n", (msg));                                                     \
        }                                                                                          \
    } while (0)

static char *dup_buf(const std::string &s) {
    char *p = static_cast<char *>(std::malloc(s.size()));
    std::memcpy(p, s.data(), s.size());
    return p;
}

static bool buf_eq(const char *p, size_t n, const std::string &s) {
    return p != nullptr && n == s.size() && std::memcmp(p, s.data(), n) == 0;
}

int main(int argc, char **argv) {
    const std::string dir = (argc > 1) ? argv[1] : "/tmp/kv_pool_test_dir";
    const std::string db_path = (argc > 2) ? argv[2] : "/tmp/kv_pool_test.db";

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::remove(db_path.c_str());
    std::remove((db_path + "-wal").c_str());
    std::remove((db_path + "-shm").c_str());

    {
        kv_pool::Storage storage(dir);
        const std::string label = "kv:abc123:0";
        const std::string payload = "hello-storage";
        CHECK(storage.save(label, payload.data(), payload.size()), "storage.save() ok");
        auto [buf, n] = storage.load(label);
        CHECK(buf_eq(buf, n, payload), "storage.load() returns saved bytes");
        std::free(buf);
    }

    {
        kv_pool::Mem mem(0);
        const std::vector<std::string> keys = {"kv:0:k", "kv:0:v", "kv:1:k"};
        const std::string d0 = "AAAA", d1 = "BBBBBB", d2 = "CC";

        std::vector<char *> ptrs = {dup_buf(d0), dup_buf(d1), dup_buf(d2)};
        std::vector<size_t> sizes = {d0.size(), d1.size(), d2.size()};
        mem.put(keys, std::move(ptrs), sizes);

        CHECK(mem.size() == 3, "size() == 3 after put");
        CHECK(mem.group_count() == 2, "group_count() == 2 (kv:0, kv:1)");
        CHECK(mem.current_bytes() == d0.size() + d1.size() + d2.size(),
              "current_bytes() sums buffers");

        auto got = mem.get(keys);
        CHECK(buf_eq(got[0].first, got[0].second, d0), "get() key0 matches");
        CHECK(buf_eq(got[1].first, got[1].second, d1), "get() key1 matches");
        CHECK(buf_eq(got[2].first, got[2].second, d2), "get() key2 matches");
        CHECK(mem.hits() == 3, "hits() == 3 after 3 gets");

        auto present = mem.has({"kv:0:k", "kv:9:x"});
        CHECK(present.size() == 2 && present[0] && !present[1], "has() reports presence");

        size_t freed = mem.remove("kv:0:k");
        CHECK(freed == d0.size() + d1.size(), "remove() frees whole group bytes");
        CHECK(mem.group_count() == 1, "group_count() == 1 after remove");
        CHECK(mem.size() == 1, "size() == 1 after remove");

        auto miss = mem.get({"kv:0:k"});
        CHECK(miss[0].first == nullptr, "get() after remove is a miss");
        CHECK(mem.misses() >= 1, "misses() incremented");
    }

    {
        kv_pool::Mem mem(0);
        std::vector<std::string> keys;
        std::vector<char *> ptrs;
        std::vector<size_t> sizes;
        for (int i = 0; i < 4; ++i) {
            keys.push_back("kv:" + std::to_string(i) + ":k");
            std::string d(100, 'x');
            ptrs.push_back(dup_buf(d));
            sizes.push_back(d.size());
        }
        mem.put(keys, std::move(ptrs), sizes);
        CHECK(mem.current_bytes() == 400, "current_bytes() == 400 before evict");

        size_t evicted = mem.evict_to_size(150);
        CHECK(evicted >= 250, "evict_to_size() freed oldest groups");
        CHECK(mem.current_bytes() <= 150, "current_bytes() <= target after evict");
        CHECK(mem.evictions() >= 1, "evictions() incremented");
    }

    {
        kv_pool::Storage storage(dir);
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(0, &storage, &record);

        const std::vector<std::string> keys = {"kv:aa:k", "kv:bb:k"};
        const std::string da = "payload-A", db = "payload-B";
        std::vector<char *> ptrs = {dup_buf(da), dup_buf(db)};
        std::vector<size_t> sizes = {da.size(), db.size()};

        record.submit("kv", {"aa", "bb"});
        record.sync();
        mem.put(keys, std::move(ptrs), sizes);
        CHECK(mem.unpersisted_count() == 2, "unpersisted_count() == 2 after put");

        auto persisted = mem.persist_groups(10);
        CHECK(persisted.size() == 2, "persist_groups() persisted 2 groups");
        CHECK(mem.unpersisted_count() == 0, "unpersisted_count() == 0 after persist");

        size_t evicted = mem.force_evict_to_size(0);
        CHECK(evicted == da.size() + db.size(), "force_evict_to_size(0) cleared memory");
        CHECK(mem.group_count() == 0, "group_count() == 0 after full evict");

        auto reloaded = mem.get(keys, true);
        CHECK(buf_eq(reloaded[0].first, reloaded[0].second, da), "get() reloads key0 from storage");
        CHECK(buf_eq(reloaded[1].first, reloaded[1].second, db), "get() reloads key1 from storage");
        CHECK(mem.hits_in_storage() == 2, "hits_in_storage() == 2 after reload");
    }

    // Test multi-layer group file aggregation, whole-group restore on first layer read,
    // expected_group_size completeness guard, and per-group reader pins.
    {
        std::filesystem::remove_all(dir, ec);
        std::remove(db_path.c_str());
        std::remove((db_path + "-wal").c_str());
        std::remove((db_path + "-shm").c_str());

        kv_pool::Storage storage(dir);
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(0, &storage, &record);
        mem.set_expected_group_size(3);

        // Put only 2 of 3 layers for group g1: should NOT be eligible for auto-spill persistence yet
        record.submit("kv", {"g1"});
        record.sync();
        mem.mark_groups_complete("kv", {"g1"});
        {
            std::vector<std::string> k12 = {"kv:g1:l0", "kv:g1:l1"};
            std::vector<char *> p12 = {dup_buf("L0_DATA"), dup_buf("L1_DATA")};
            std::vector<size_t> s12 = {7, 7};
            mem.put(k12, std::move(p12), s12);
        }
        auto early_spill = mem.spill_groups(1, /*evict_only_override=*/0);
        CHECK(early_spill.persisted_groups == 0 && early_spill.evicted_groups == 0,
              "incomplete group (2/3 layers) is not persisted prematurely");

        // Now put the 3rd layer: group g1 becomes complete (3/3 layers)
        {
            std::vector<std::string> k3 = {"kv:g1:l2"};
            std::vector<char *> p3 = {dup_buf("L2_DATA_LONG")};
            std::vector<size_t> s3 = {12};
            mem.put(k3, std::move(p3), s3);
        }

        // Test per-group reader pin blocking eviction of pinned group while allowing duplicate put
        auto l0_got = mem.get({"kv:g1:l0"}, false, /*caller_holds_deletion_guard=*/true);
        CHECK(buf_eq(l0_got[0].first, l0_got[0].second, "L0_DATA"), "pinned get l0 ok");
        {
            // Idempotent duplicate put while pinned
            std::vector<std::string> dup_k = {"kv:g1:l0"};
            std::vector<char *> dup_p = {dup_buf("L0_DATA")};
            std::vector<size_t> dup_s = {7};
            mem.put(dup_k, std::move(dup_p), dup_s);
            CHECK(buf_eq(l0_got[0].first, l0_got[0].second, "L0_DATA"),
                  "pinned pointer remains valid across duplicate put");
        }
        mem.release_group_guards({"kv:g1:l0"});

        // Spill complete group g1 to single aggregated .grp.bin file and evict from DDR
        auto full_spill = mem.spill_groups(1, /*evict_only_override=*/0);
        CHECK(full_spill.persisted_groups == 1 && full_spill.evicted_groups == 1,
              "complete group (3/3 layers) persisted and evicted in batch");
        CHECK(mem.group_count() == 0, "group evicted from DDR after spill");

        mem.reset_stats();
        // Request Layer 0: should load the entire group (l0, l1, l2) from the single .grp.bin file
        auto r_l0 = mem.get({"kv:g1:l0"}, true, /*caller_holds_deletion_guard=*/true);
        CHECK(buf_eq(r_l0[0].first, r_l0[0].second, "L0_DATA"), "reload l0 from group file");
        CHECK(mem.hits_in_storage() == 1 && mem.hits() == 0,
              "hits_in_storage == 1 after l0 reload");
        CHECK(mem.size() == 3, "all 3 layers restored into DDR on l0 reload");
        mem.release_group_guards({"kv:g1:l0"});

        // Subsequent Layer 1 and Layer 2 gets hit in memory without disk I/O and count as hits_in_storage
        auto r_l1 = mem.get({"kv:g1:l1"}, true, /*caller_holds_deletion_guard=*/true);
        auto r_l2 = mem.get({"kv:g1:l2"}, true, /*caller_holds_deletion_guard=*/true);
        CHECK(buf_eq(r_l1[0].first, r_l1[0].second, "L1_DATA"), "l1 hit in DDR after group reload");
        CHECK(buf_eq(r_l2[0].first, r_l2[0].second, "L2_DATA_LONG"),
              "l2 hit in DDR after group reload");
        CHECK(mem.hits_in_storage() == 3 && mem.hits() == 0,
              "hits_in_storage == 3 across all 3 restored layers");
        mem.release_group_guards({"kv:g1:l1"});
        mem.release_group_guards({"kv:g1:l2"});

        // Second revisit in DDR counts as hits()
        auto r_again = mem.get({"kv:g1:l0", "kv:g1:l1", "kv:g1:l2"}, true);
        CHECK(mem.hits() == 3 && mem.hits_in_storage() == 3,
              "second revisit in DDR increments hits()");

        // Test overwrite with different payload while group is pinned (retired_buffers safety)
        auto pinned = mem.get({"kv:g1:l0"}, false, /*caller_holds_deletion_guard=*/true);
        const char *old_ptr = pinned[0].first;
        {
            std::vector<std::string> ow_k = {"kv:g1:l0"};
            std::vector<char *> ow_p = {dup_buf("L0_NEW!")};
            std::vector<size_t> ow_s = {7};
            mem.put(ow_k, std::move(ow_p), ow_s);
        }
        // Old pointer must still be readable while pin is held!
        CHECK(buf_eq(old_ptr, 7, "L0_DATA"), "old buffer kept alive in retired_buffers while pinned");
        mem.release_group_guards({"kv:g1:l0"});
        auto updated = mem.get({"kv:g1:l0"}, false);
        CHECK(buf_eq(updated[0].first, updated[0].second, "L0_NEW!"),
              "new buffer visible after overwrite");
    }

    // Test background auto-spill pre-persist (at low watermark) + fast-path eviction (at high watermark)
    {
        std::filesystem::remove_all(dir, ec);
        std::remove(db_path.c_str());
        std::remove((db_path + "-wal").c_str());
        std::remove((db_path + "-shm").c_str());

        kv_pool::Storage storage(dir);
        kv_pool::Record record(db_path, true);
        // Capacity = 1000 bytes, low_wm = 600 bytes (60%), high_wm = 800 bytes (80%)
        kv_pool::Mem mem(1000, &storage, &record);
        mem.set_expected_group_size(2);
        mem.configure_auto_spill(/*enabled=*/true, /*evict_only=*/false,
                                 /*high_watermark=*/0.80, /*low_watermark=*/0.60,
                                 /*batch_groups=*/4, /*pace_ms=*/0, /*evict_grace_ms=*/10);

        // Insert 4 complete groups of 160 bytes each = 640 bytes (>= 600 low_wm, < 800 high_wm)
        for (int i = 0; i < 4; ++i) {
            std::string cid = "c" + std::to_string(i);
            record.submit("kv", {cid});
            std::vector<std::string> ks = {"kv:" + cid + ":l0", "kv:" + cid + ":l1"};
            std::vector<char *> ps = {dup_buf(std::string(80, 'a' + i)),
                                      dup_buf(std::string(80, 'A' + i))};
            std::vector<size_t> ss = {80, 80};
            mem.put(ks, std::move(ps), ss);
        }
        record.sync();

        // Wait briefly for background pre-persist to persist all 4 groups without evicting them
        for (int wait_i = 0; wait_i < 100 && mem.unpersisted_count() > 0; ++wait_i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        CHECK(mem.unpersisted_count() == 0,
              "background pre-persist persisted groups between low and high watermarks");
        CHECK(mem.group_count() == 4,
              "pre-persisted groups remain in DDR below high watermark");

        // Now insert 2 more groups (total 960 bytes >= 800 high_wm) -> triggers fast-path eviction
        for (int i = 4; i < 6; ++i) {
            std::string cid = "c" + std::to_string(i);
            record.submit("kv", {cid});
            std::vector<std::string> ks = {"kv:" + cid + ":l0", "kv:" + cid + ":l1"};
            std::vector<char *> ps = {dup_buf(std::string(80, 'a' + i)),
                                      dup_buf(std::string(80, 'A' + i))};
            std::vector<size_t> ss = {80, 80};
            mem.put(ks, std::move(ps), ss);
        }
        record.sync();

        for (int wait_i = 0; wait_i < 100 && mem.current_bytes() > 600; ++wait_i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        CHECK(mem.current_bytes() <= 600,
              "auto-spill fast-evicted pre-persisted groups down to low watermark");
        auto st = mem.auto_spill_status();
        CHECK(st.auto_persisted_groups >= 4 && st.auto_evicted_groups >= 2,
              "auto_spill status tracks persisted and evicted groups");
        mem.stop_background_workers();
    }

    std::filesystem::remove_all(dir, ec);
    std::remove(db_path.c_str());
    std::remove((db_path + "-wal").c_str());
    std::remove((db_path + "-shm").c_str());

    if (g_failures == 0) {
        std::printf("\nAll kv_pool checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d kv_pool check(s) failed.\n", g_failures);
    return 1;
}
