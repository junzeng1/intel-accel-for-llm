// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "iaxl_common.h"
#include "env.h"
#include "kv_pool.h"
#include "task_queue.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kv_pool {

namespace {

double now_timestamp() {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

struct SubEntry {
    char *data = nullptr;
    size_t size = 0;

    ~SubEntry() {
        if (data) {
            free(data);
            data = nullptr;
        }
    }

    SubEntry(const SubEntry &) = delete;
    SubEntry &operator=(const SubEntry &) = delete;

    SubEntry() = default;

    SubEntry(SubEntry &&other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }

    SubEntry &operator=(SubEntry &&other) noexcept {
        if (this != &other) {
            if (data)
                free(data);
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
};

struct EntryGroup {
    EntryGroup *lru_prev = nullptr;
    EntryGroup *lru_next = nullptr;

    EntryGroup *unpersisted_prev = nullptr;
    EntryGroup *unpersisted_next = nullptr;
    bool is_persisted = false;
    bool is_persisting = false;
    bool is_unpublishing = false;
    bool is_complete = false;

    uint64_t version = 0;
    uint64_t access_seq = 0;

    const std::string *group_key_ptr = nullptr;

    double created_at = 0.0;
    double last_access = 0.0;

    std::unordered_map<std::string, SubEntry> entries;

    size_t total_bytes() const {
        size_t total = 0;
        for (const auto &[_, sub] : entries)
            total += sub.size;
        return total;
    }

    size_t entry_count() const { return entries.size(); }

    SubEntry *find(std::string_view tensor_key) {
        auto it = entries.find(std::string(tensor_key));
        return it != entries.end() ? &it->second : nullptr;
    }

    const SubEntry *find(std::string_view tensor_key) const {
        auto it = entries.find(std::string(tensor_key));
        return it != entries.end() ? &it->second : nullptr;
    }

    SubEntry &get_or_create(const std::string &tensor_key) { return entries[tensor_key]; }

    ~EntryGroup() = default;

    EntryGroup() { entries.reserve((size_t)envs.IAXL_CACHE_CACHEGROUP_SIZE); }

    EntryGroup(const EntryGroup &) = delete;
    EntryGroup &operator=(const EntryGroup &) = delete;

    EntryGroup(EntryGroup &&other) noexcept
        : lru_prev(other.lru_prev), lru_next(other.lru_next),
          unpersisted_prev(other.unpersisted_prev), unpersisted_next(other.unpersisted_next),
          is_persisted(other.is_persisted), is_persisting(other.is_persisting),
          is_unpublishing(other.is_unpublishing), is_complete(other.is_complete),
          version(other.version), access_seq(other.access_seq),
          group_key_ptr(other.group_key_ptr), created_at(other.created_at),
          last_access(other.last_access), entries(std::move(other.entries)) {
        other.lru_prev = nullptr;
        other.lru_next = nullptr;
        other.unpersisted_prev = nullptr;
        other.unpersisted_next = nullptr;
        other.group_key_ptr = nullptr;
    }

    EntryGroup &operator=(EntryGroup &&other) noexcept {
        if (this != &other) {
            lru_prev = other.lru_prev;
            lru_next = other.lru_next;
            unpersisted_prev = other.unpersisted_prev;
            unpersisted_next = other.unpersisted_next;
            is_persisted = other.is_persisted;
            is_persisting = other.is_persisting;
            is_unpublishing = other.is_unpublishing;
            is_complete = other.is_complete;
            version = other.version;
            access_seq = other.access_seq;
            group_key_ptr = other.group_key_ptr;
            created_at = other.created_at;
            last_access = other.last_access;
            entries = std::move(other.entries);
            other.lru_prev = nullptr;
            other.lru_next = nullptr;
            other.unpersisted_prev = nullptr;
            other.unpersisted_next = nullptr;
            other.group_key_ptr = nullptr;
        }
        return *this;
    }
};

} // namespace

class Mem::Impl {
  public:
    explicit Impl(size_t capacity_bytes, Storage *storage, Record *record)
        : capacity_bytes_(capacity_bytes), current_bytes_(0), total_entries_(0),
          evict_enabled_(true), storage_(storage), record_(record) {
        cache_.reserve((size_t)envs.IAXL_CACHE_CACHEGROUP_NUM);

        lru_head_.lru_next = &lru_head_;
        lru_head_.lru_prev = &lru_head_;

        unpersisted_head_.unpersisted_next = &unpersisted_head_;
        unpersisted_head_.unpersisted_prev = &unpersisted_head_;

        configure_auto_spill(envs.IAXL_CACHE_AUTO_SPILL_ENABLE,
                             envs.IAXL_CACHE_AUTO_SPILL_EVICT_ONLY, envs.IAXL_CACHE_HIGH_WATERMARK,
                             envs.IAXL_CACHE_LOW_WATERMARK,
                             (size_t)std::max(1, envs.IAXL_CACHE_SPILL_BATCH_GROUPS),
                             (size_t)std::max(0, envs.IAXL_CACHE_SPILL_PACE_MS),
                             (size_t)std::max(0, envs.IAXL_CACHE_EVICT_GRACE_MS));
    }

    ~Impl() {
        stop_background_workers();
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }

    void stop_background_workers() {
        {
            std::lock_guard<std::mutex> lk(spill_cv_mutex_);
            spill_stop_.store(true, std::memory_order_release);
        }
        spill_cv_.notify_all();
        if (spill_worker_.joinable()) {
            spill_worker_.join();
        }
        spill_running_.store(false, std::memory_order_release);
    }

    void configure_auto_spill(bool enabled, bool evict_only, double high_watermark,
                              double low_watermark, size_t batch_groups, size_t pace_ms,
                              size_t evict_grace_ms) {
        if (high_watermark > 1.0 && high_watermark <= 100.0)
            high_watermark /= 100.0;
        if (low_watermark > 1.0 && low_watermark <= 100.0)
            low_watermark /= 100.0;
        if (high_watermark <= 0.0 || high_watermark > 1.0)
            high_watermark = 0.80;
        if (low_watermark < 0.0 || low_watermark >= high_watermark)
            low_watermark = std::max(0.0, high_watermark - 0.05);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            high_watermark_ = high_watermark;
            low_watermark_ = low_watermark;
            spill_batch_groups_ = std::max<size_t>(1, batch_groups);
            spill_pace_ms_ = pace_ms;
            evict_grace_ms_ = evict_grace_ms;
            recompute_watermark_bytes_unlocked();

            bool can_enable = enabled && capacity_bytes_ > 0 && (evict_only || storage_ != nullptr);
            auto_spill_evict_only_.store(evict_only, std::memory_order_release);
            auto_spill_enabled_.store(can_enable, std::memory_order_release);
        }

        if (auto_spill_enabled_.load(std::memory_order_acquire)) {
            ensure_spill_worker_started();
            maybe_notify_spill();
        }
    }

    AutoSpillStatus auto_spill_status() const {
        AutoSpillStatus st;
        std::lock_guard<std::mutex> lock(mutex_);
        st.enabled = auto_spill_enabled_.load(std::memory_order_acquire);
        st.evict_only = auto_spill_evict_only_.load(std::memory_order_acquire);
        st.running = spill_running_.load(std::memory_order_acquire);
        st.high_watermark = high_watermark_;
        st.low_watermark = low_watermark_;
        st.high_watermark_bytes = high_watermark_bytes_;
        st.low_watermark_bytes = low_watermark_bytes_;
        st.batch_groups = spill_batch_groups_;
        st.pace_ms = spill_pace_ms_;
        st.evict_grace_ms = evict_grace_ms_;
        st.spill_rounds = spill_rounds_.load(std::memory_order_relaxed);
        st.auto_persisted_groups = auto_persisted_groups_.load(std::memory_order_relaxed);
        st.auto_persisted_bytes = auto_persisted_bytes_.load(std::memory_order_relaxed);
        st.auto_evicted_groups = auto_evicted_groups_.load(std::memory_order_relaxed);
        st.auto_evicted_bytes = auto_evicted_bytes_.load(std::memory_order_relaxed);
        return st;
    }

    void mark_groups_complete(const std::string &label,
                              const std::vector<std::string> &chunk_ids) {
        auto group_keys = make_group_keys(label, chunk_ids);
        std::lock_guard<std::mutex> lock(mutex_);
        has_complete_tracking_ = true;
        for (const auto &gk : group_keys) {
            auto it = cache_.find(gk);
            if (it != cache_.end()) {
                it->second.is_complete = true;
            } else {
                pre_completed_groups_.insert(gk);
            }
        }
    }

    void lock() { mutex_.lock(); }
    void unlock() { mutex_.unlock(); }

    void acquire_deletion_guard() { deletion_guard_count_.fetch_add(1, std::memory_order_acq_rel); }

    void release_deletion_guard() {
        if (deletion_guard_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(deletion_cv_mutex_);
            deletion_cv_.notify_all();
        }
    }

    bool is_deletion_guarded() const {
        return deletion_guard_count_.load(std::memory_order_acquire) > 0;
    }

    void wait_deletion_guard() {
        if (deletion_guard_count_.load(std::memory_order_acquire) == 0)
            return;
        std::unique_lock<std::mutex> lk(deletion_cv_mutex_);
        deletion_cv_.wait(
            lk, [this] { return deletion_guard_count_.load(std::memory_order_acquire) == 0; });
    }

    void put(const std::vector<std::string> &keys, std::vector<char *> &&data_ptrs,
             const std::vector<size_t> &sizes, const std::vector<size_t> &unzip_sizes) {
        IAXL_CHECK(keys.size() == data_ptrs.size(),
                   "Mem::put: keys and data pointers must have the same length");
        IAXL_CHECK(keys.size() == sizes.size(),
                   "Mem::put: keys and sizes must have the same length");
        IAXL_CHECK(unzip_sizes.empty() || keys.size() == unzip_sizes.size(),
                   "Mem::put: unzip sizes must be empty or match keys length");
        bool should_notify = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            put_unlocked(keys, std::move(data_ptrs), sizes, unzip_sizes);
            if (auto_spill_enabled_.load(std::memory_order_relaxed) && high_watermark_bytes_ > 0 &&
                current_bytes_ >= high_watermark_bytes_ &&
                !spill_running_.load(std::memory_order_relaxed)) {
                should_notify = true;
            }
        }
        if (should_notify) {
            maybe_notify_spill();
        }
    }

    std::vector<std::pair<const char *, size_t>> get(const std::vector<std::string> &keys,
                                                     bool allow_using_omp,
                                                     bool caller_holds_deletion_guard) {
        struct ActiveGetGuard {
            std::atomic<int> &counter;
            explicit ActiveGetGuard(std::atomic<int> &c) : counter(c) {
                counter.fetch_add(1, std::memory_order_relaxed);
            }
            ~ActiveGetGuard() { counter.fetch_sub(1, std::memory_order_relaxed); }
        } active_guard(active_gets_);

        const size_t n = keys.size();
        std::vector<std::pair<const char *, size_t>> results(n, {nullptr, 0});
        double now = now_timestamp();

        // Phase 1: fast in-memory lookup under lock
        std::vector<size_t> miss_indices;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            miss_indices.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                results[i] = get_one_unlocked(keys[i], now);
                if (results[i].first == nullptr) {
                    miss_indices.push_back(i);
                }
            }
        }

        if (miss_indices.empty()) {
            return results;
        }

        if (!storage_ || !record_) {
            misses_.fetch_add(miss_indices.size(), std::memory_order_relaxed);
            return results;
        }

        // Release caller's deletion guard during slow storage I/O so background eviction
        // of unrelated cold groups is not blocked while reading from disk.
        if (caller_holds_deletion_guard) {
            release_deletion_guard();
        }

        struct GuardRestorer {
            Impl *self;
            bool active;
            ~GuardRestorer() {
                if (active)
                    self->acquire_deletion_guard();
            }
        } restorer{this, caller_holds_deletion_guard};

        // Phase 2: lock-free Record check + Storage batch load
        std::vector<std::string> check_keys;
        check_keys.reserve(miss_indices.size());
        for (size_t idx : miss_indices) {
            auto [group_key_sv, _] = parse_full_label(keys[idx]);
            check_keys.emplace_back(group_key_sv);
        }

        std::vector<bool> is_persisted = record_->is_persisted(check_keys);
        std::vector<size_t> storage_indices;
        std::vector<std::string> load_keys;
        storage_indices.reserve(miss_indices.size());
        load_keys.reserve(miss_indices.size());

        for (size_t i = 0; i < miss_indices.size(); ++i) {
            if (is_persisted[i]) {
                storage_indices.push_back(miss_indices[i]);
                load_keys.push_back(keys[miss_indices[i]]);
            }
        }

        misses_.fetch_add(miss_indices.size() - storage_indices.size(), std::memory_order_relaxed);

        std::vector<std::pair<char *, size_t>> loaded_buffers;
        if (!storage_indices.empty()) {
            loaded_buffers = storage_->load_batch(load_keys, allow_using_omp);
        }

        // Re-acquire deletion guard BEFORE locking mutex_ and resolving live pointers
        if (restorer.active) {
            acquire_deletion_guard();
            restorer.active = false;
        }

        // Phase 3: insert loaded buffers and resolve live pointers under lock
        bool should_notify = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t i = 0; i < storage_indices.size(); ++i) {
                size_t key_index = storage_indices[i];
                char *buf = loaded_buffers[i].first;
                size_t sz = loaded_buffers[i].second;
                if (buf) {
                    put_loaded_one_unlocked(keys[key_index], buf, sz, now);
                    hits_in_storage_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    misses_.fetch_add(1, std::memory_order_relaxed);
                }
            }

            // Resolve all pointers under the active deletion guard
            for (size_t i = 0; i < n; ++i) {
                auto [group_key_sv, tensor_key_sv] = parse_full_label(keys[i]);
                auto it = cache_.find(std::string(group_key_sv));
                if (it != cache_.end()) {
                    SubEntry *sub = it->second.find(tensor_key_sv);
                    if (sub && sub->data) {
                        results[i] = {sub->data, sub->size};
                        continue;
                    }
                }
                results[i] = {nullptr, 0};
            }

            if (auto_spill_enabled_.load(std::memory_order_relaxed) && high_watermark_bytes_ > 0 &&
                current_bytes_ >= high_watermark_bytes_ &&
                !spill_running_.load(std::memory_order_relaxed)) {
                should_notify = true;
            }
        }

        if (should_notify) {
            maybe_notify_spill();
        }

        return results;
    }

    std::vector<bool> has(const std::vector<std::string> &keys) {
        std::lock_guard<std::mutex> lock(mutex_);
        return has_unlocked(keys);
    }

    std::vector<std::tuple<std::string, size_t>> persist_groups(size_t max_groups) {
        if (!storage_) {
            std::cerr << "\033[1;31m*** FATAL: No storage backend configured, cannot persist "
                         "groups.\033[0m"
                      << std::endl;
            std::abort();
        }
        if (max_groups == 0) {
            return {};
        }

        struct GroupSnap {
            std::string group_key;
            uint64_t version;
            std::vector<std::pair<std::string, std::vector<char>>> entries;
        };

        // Phase 1: snapshot unpersisted groups under lock
        std::vector<GroupSnap> group_snaps;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            EntryGroup *group = unpersisted_head_.unpersisted_next;
            while (group != &unpersisted_head_ && group_snaps.size() < max_groups) {
                EntryGroup *next = group->unpersisted_next;
                if (!group->is_persisting && !group->is_unpublishing && !group->entries.empty()) {
                    group->is_persisting = true;
                    GroupSnap gsnap;
                    gsnap.group_key = *group->group_key_ptr;
                    gsnap.version = group->version;
                    gsnap.entries.reserve(group->entries.size());
                    for (const auto &[tensor_key, sub] : group->entries) {
                        if (sub.data && sub.size > 0) {
                            std::string full_label = make_full_label(gsnap.group_key, tensor_key);
                            gsnap.entries.emplace_back(
                                std::move(full_label),
                                std::vector<char>(sub.data, sub.data + sub.size));
                        }
                    }
                    group_snaps.push_back(std::move(gsnap));
                }
                group = next;
            }
        }

        if (group_snaps.empty()) {
            return {};
        }

        // Phase 2: write to Storage outside lock
        std::vector<std::string> succeeded_groups;
        std::unordered_map<std::string, size_t> group_bytes;
        succeeded_groups.reserve(group_snaps.size());

        for (const auto &gsnap : group_snaps) {
            std::vector<StorageBackend::WriteItem> write_items;
            write_items.reserve(gsnap.entries.size());
            size_t total_sz = 0;
            for (const auto &[fl, buf] : gsnap.entries) {
                write_items.push_back({fl, buf.data(), buf.size()});
                total_sz += buf.size();
            }
            auto res = storage_->save_batch(write_items);
            bool all_ok = !res.empty();
            for (bool ok : res) {
                if (!ok) {
                    all_ok = false;
                    break;
                }
            }
            if (all_ok) {
                succeeded_groups.push_back(gsnap.group_key);
                group_bytes[gsnap.group_key] = total_sz;
            }
        }

        if (record_ && !succeeded_groups.empty()) {
            record_->mark_persisted(succeeded_groups);
        }

        // Phase 3: update in-memory state with version check under lock
        std::unordered_set<std::string> succeeded_set(succeeded_groups.begin(),
                                                      succeeded_groups.end());
        std::vector<std::tuple<std::string, size_t>> persisted_groups;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto &gsnap : group_snaps) {
                auto it = cache_.find(gsnap.group_key);
                if (it == cache_.end())
                    continue;
                EntryGroup &group = it->second;
                group.is_persisting = false;
                if (succeeded_set.count(gsnap.group_key) > 0) {
                    if (group.version == gsnap.version && !group.is_persisted) {
                        group.is_persisted = true;
                        unpersisted_remove(&group);
                    }
                    persisted_groups.emplace_back(gsnap.group_key, group_bytes[gsnap.group_key]);
                }
            }
        }

        return persisted_groups;
    }

    std::vector<std::tuple<std::string, size_t>> evict_groups(size_t max_groups) {
        if (!evict_enabled_.load(std::memory_order_acquire) || max_groups == 0) {
            return {};
        }

        wait_deletion_guard();

        struct EvictCandidate {
            std::string group_key;
            std::string rep_full_label;
            uint64_t version;
            uint64_t access_seq;
            size_t bytes;
            bool is_persisted;
        };

        // Phase 1: select oldest LRU groups under lock
        std::vector<EvictCandidate> candidates;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            EntryGroup *group = lru_head_.lru_next;
            while (group != &lru_head_ && candidates.size() < max_groups) {
                EntryGroup *next = group->lru_next;
                if (!group->is_persisting && !group->is_unpublishing && !group->entries.empty()) {
                    group->is_unpublishing = true;
                    const std::string &gk = *group->group_key_ptr;
                    const std::string &tk = group->entries.begin()->first;
                    candidates.push_back({gk, make_full_label(gk, tk), group->version,
                                          group->access_seq, group->total_bytes(),
                                          group->is_persisted});
                }
                group = next;
            }
        }

        if (candidates.empty()) {
            return {};
        }

        // Phase 2: check and remove unpersisted groups from Record outside lock
        if (record_) {
            std::vector<std::string> check_keys;
            for (const auto &c : candidates) {
                if (!c.is_persisted) {
                    check_keys.push_back(c.group_key);
                }
            }
            if (!check_keys.empty()) {
                std::vector<bool> db_persisted = record_->is_persisted(check_keys);
                std::vector<std::string> to_remove;
                for (size_t i = 0; i < check_keys.size(); ++i) {
                    if (!db_persisted[i]) {
                        to_remove.push_back(check_keys[i]);
                    }
                }
                if (!to_remove.empty()) {
                    record_->remove(to_remove);
                }
            }
        }

        wait_deletion_guard();

        // Phase 3: remove from cache under lock if not concurrently modified/accessed
        std::vector<std::tuple<std::string, size_t>> evicted_groups;
        std::unordered_map<std::string, std::vector<std::string>> resubmit_by_label;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto &c : candidates) {
                auto it = cache_.find(c.group_key);
                if (it == cache_.end())
                    continue;
                EntryGroup &group = it->second;
                group.is_unpublishing = false;
                if (!is_deletion_guarded() && group.version == c.version &&
                    group.access_seq == c.access_seq) {
                    size_t freed = remove_unlocked(c.rep_full_label);
                    evictions_.fetch_add(1, std::memory_order_relaxed);
                    evicted_groups.emplace_back(c.group_key, freed);
                } else if (record_ && !c.is_persisted && !group.is_persisted) {
                    auto [label_sv, chunk_sv] = parse_group_key(c.group_key);
                    resubmit_by_label[std::string(label_sv)].emplace_back(chunk_sv);
                }
            }
        }

        if (record_) {
            for (const auto &[label, chunks] : resubmit_by_label) {
                record_->submit(label, chunks);
            }
        }

        return evicted_groups;
    }

    SpillResult spill_groups(size_t max_groups, int evict_only_override = -1) {
        std::lock_guard<std::mutex> op_lk(spill_op_mutex_);
        bool evict_only = (evict_only_override >= 0)
                              ? (evict_only_override != 0)
                              : auto_spill_evict_only_.load(std::memory_order_acquire);
        return run_spill_loop(/*target_bytes=*/0, /*use_target_bytes=*/false, max_groups,
                              evict_only, /*is_auto_worker=*/false);
    }

    SpillResult spill_to_size(size_t target_bytes, size_t max_groups = 0,
                              int evict_only_override = -1) {
        std::lock_guard<std::mutex> op_lk(spill_op_mutex_);
        bool evict_only = (evict_only_override >= 0)
                              ? (evict_only_override != 0)
                              : auto_spill_evict_only_.load(std::memory_order_acquire);
        return run_spill_loop(target_bytes, /*use_target_bytes=*/true, max_groups, evict_only,
                              /*is_auto_worker=*/false);
    }

    void put_unlocked(const std::vector<std::string> &keys, std::vector<char *> &&data_ptrs,
                      const std::vector<size_t> &sizes, const std::vector<size_t> &unzip_sizes) {
        double now = now_timestamp();
        for (size_t i = 0; i < keys.size(); i++) {

            size_t unzip_size = (i < unzip_sizes.size()) ? unzip_sizes[i] : 0;
            put_one_unlocked(keys[i], data_ptrs[i], sizes[i], unzip_size, now);
            data_ptrs[i] = nullptr;
        }
    }

    std::vector<std::pair<const char *, size_t>> get_unlocked(const std::vector<std::string> &keys,
                                                              bool allow_using_omp) {
        std::vector<std::pair<const char *, size_t>> results(keys.size());
        double now = now_timestamp();

        std::vector<size_t> miss_indices;
        miss_indices.reserve(keys.size());

        for (size_t i = 0; i < keys.size(); ++i) {
            results[i] = get_one_unlocked(keys[i], now);
            if (results[i].first == nullptr) {
                miss_indices.push_back(i);
            }
        }

        if (miss_indices.empty()) {
            return results;
        }

        if (!storage_) {
            misses_.fetch_add(miss_indices.size(), std::memory_order_relaxed);
            return results;
        }

        std::vector<size_t> storage_indices;
        storage_indices.reserve(miss_indices.size());

        if (record_) {
            std::vector<std::string> check_keys;
            check_keys.reserve(miss_indices.size());
            for (size_t idx : miss_indices) {
                auto [group_key_sv, _] = parse_full_label(keys[idx]);
                check_keys.emplace_back(group_key_sv);
            }

            std::vector<bool> is_persisted = record_->is_persisted(check_keys);

            for (size_t i = 0; i < miss_indices.size(); ++i) {
                if (is_persisted[i]) {
                    storage_indices.push_back(miss_indices[i]);
                } else {
                }
            }
            misses_.fetch_add(miss_indices.size() - storage_indices.size(),
                              std::memory_order_relaxed);
        } else {

            misses_.fetch_add(miss_indices.size(), std::memory_order_relaxed);
            return results;
        }

        if (storage_indices.empty()) {
            return results;
        }

        struct LoadedData {
            size_t index;
            char *buffer;
            size_t size;
        };
        std::vector<LoadedData> loaded_data(storage_indices.size());

        if (allow_using_omp) {
#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < storage_indices.size(); ++i) {
                size_t key_index = storage_indices[i];
                auto [buffer, size] = storage_->load(keys[key_index]);
                loaded_data[i] = {key_index, buffer, size};
            }
        } else {
            for (size_t i = 0; i < storage_indices.size(); ++i) {
                size_t key_index = storage_indices[i];
                auto [buffer, size] = storage_->load(keys[key_index]);
                loaded_data[i] = {key_index, buffer, size};
            }
        }

        for (const auto &item : loaded_data) {
            if (item.buffer) {

                put_one_unlocked(keys[item.index], item.buffer, item.size, 0, now);

                mark_persisted_unlocked({keys[item.index]});

                hits_in_storage_.fetch_add(1, std::memory_order_relaxed);

                auto [group_key_sv, tensor_key_sv] = parse_full_label(keys[item.index]);
                auto it = cache_.find(std::string(group_key_sv));
                SubEntry *sub = it->second.find(tensor_key_sv);
                results[item.index] = {sub->data, sub->size};
            } else {

                misses_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        return results;
    }

    std::vector<bool> has_unlocked(const std::vector<std::string> &keys) const {
        std::vector<bool> results;
        results.reserve(keys.size());
        for (const auto &key : keys) {
            auto [group_key_sv, tensor_key_sv] = parse_full_label(key);
            auto it = cache_.find(std::string(group_key_sv));
            if (it != cache_.end()) {
                if (tensor_key_sv.empty()) {
                    results.push_back(true);
                } else {
                    const SubEntry *sub = it->second.find(tensor_key_sv);
                    results.push_back(sub != nullptr && sub->data != nullptr);
                }
            } else {
                results.push_back(false);
            }
        }
        return results;
    }

    std::vector<std::tuple<std::string, const char *, size_t>>
    get_unpersisted_unlocked(size_t max_count) {
        std::vector<std::tuple<std::string, const char *, size_t>> results;
        EntryGroup *group = unpersisted_head_.unpersisted_next;
        size_t groups_returned = 0;
        while (group != &unpersisted_head_ && groups_returned < max_count) {
            const std::string &group_key = *group->group_key_ptr;
            for (const auto &[tensor_key, sub] : group->entries) {
                std::string full_label = make_full_label(group_key, tensor_key);
                results.emplace_back(std::move(full_label), sub.data, sub.size);
            }
            groups_returned++;
            group = group->unpersisted_next;
        }
        return results;
    }

    std::vector<std::tuple<std::string, const char *, size_t>>
    get_lru_oldest_unlocked(size_t max_count) {
        std::vector<std::tuple<std::string, const char *, size_t>> results;
        EntryGroup *group = lru_head_.lru_next;
        size_t groups_returned = 0;
        while (group != &lru_head_ && groups_returned < max_count) {
            const std::string &group_key = *group->group_key_ptr;
            for (const auto &[tensor_key, sub] : group->entries) {
                std::string full_label = make_full_label(group_key, tensor_key);
                results.emplace_back(std::move(full_label), sub.data, sub.size);
            }
            groups_returned++;
            group = group->lru_next;
        }
        return results;
    }

    void mark_persisted(const std::vector<std::string> &keys) {
        std::lock_guard<std::mutex> lock(mutex_);
        mark_persisted_unlocked(keys);
    }

    void mark_persisted_unlocked(const std::vector<std::string> &keys) {
        for (const auto &key : keys) {
            auto [group_key_sv, _] = parse_full_label(key);
            auto it = cache_.find(std::string(group_key_sv));
            if (it != cache_.end()) {
                EntryGroup &group = it->second;
                if (!group.is_persisted) {
                    group.is_persisted = true;
                    unpersisted_remove(&group);
                }
            }
        }
    }

    size_t unpersisted_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return unpersisted_count_unlocked();
    }

    size_t unpersisted_count_unlocked() const {
        size_t count = 0;
        const EntryGroup *group = unpersisted_head_.unpersisted_next;
        while (group != &unpersisted_head_) {
            count++;
            group = group->unpersisted_next;
        }
        return count;
    }

    void set_evict_enabled(bool enabled) {
        evict_enabled_.store(enabled, std::memory_order_release);
    }

    bool is_evict_enabled() const { return evict_enabled_.load(std::memory_order_acquire); }

    size_t evict_to_size(size_t target_bytes) {
        if (!evict_enabled_.load(std::memory_order_acquire)) {
            return 0;
        }
        return force_evict_to_size(target_bytes);
    }

    size_t force_evict_to_size(size_t target_bytes) {
        wait_deletion_guard();
        std::vector<std::string> unpersisted_groups_to_remove;
        size_t evicted = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            evicted = evict_unlocked(target_bytes, &unpersisted_groups_to_remove);
        }
        if (record_ && !unpersisted_groups_to_remove.empty()) {
            record_->remove(unpersisted_groups_to_remove);
        }
        return evicted;
    }

    size_t remove(const std::string &key) {
        wait_deletion_guard();
        std::lock_guard<std::mutex> lock(mutex_);
        return remove_unlocked(key);
    }

    size_t remove_unlocked(const std::string &key) {
        auto [group_key_sv, _] = parse_full_label(key);
        auto it = cache_.find(std::string(group_key_sv));
        if (it != cache_.end()) {
            EntryGroup &group = it->second;
            size_t bytes = group.total_bytes();
            current_bytes_ -= bytes;
            total_entries_ -= group.entry_count();
            lru_remove(&group);
            unpersisted_remove(&group);
            cache_.erase(it);
            return bytes;
        }
        return 0;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_entries_;
    }

    size_t group_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

    size_t current_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_bytes_;
    }

    size_t capacity_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return capacity_bytes_;
    }

    void set_capacity_bytes(size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        capacity_bytes_ = capacity;
        recompute_watermark_bytes_unlocked();
    }

    uint64_t hits() const { return hits_.load(std::memory_order_relaxed); }
    uint64_t hits_in_storage() const { return hits_in_storage_.load(std::memory_order_relaxed); }
    uint64_t misses() const { return misses_.load(std::memory_order_relaxed); }
    uint64_t puts() const { return puts_.load(std::memory_order_relaxed); }
    uint64_t evictions() const { return evictions_.load(std::memory_order_relaxed); }

    uint64_t total_unzip_bytes() const {
        return total_unzip_bytes_.load(std::memory_order_relaxed);
    }
    uint64_t total_zip_bytes() const { return total_zip_bytes_.load(std::memory_order_relaxed); }

    double compression_ratio() const {
        uint64_t unzip = total_unzip_bytes_.load(std::memory_order_relaxed);
        uint64_t zip = total_zip_bytes_.load(std::memory_order_relaxed);
        return (zip > 0) ? static_cast<double>(unzip) / static_cast<double>(zip) : 0.0;
    }

    void reset_stats() {
        hits_.store(0, std::memory_order_relaxed);
        hits_in_storage_.store(0, std::memory_order_relaxed);
        misses_.store(0, std::memory_order_relaxed);
        puts_.store(0, std::memory_order_relaxed);
        evictions_.store(0, std::memory_order_relaxed);
        total_unzip_bytes_.store(0, std::memory_order_relaxed);
        total_zip_bytes_.store(0, std::memory_order_relaxed);
        spill_rounds_.store(0, std::memory_order_relaxed);
        auto_persisted_groups_.store(0, std::memory_order_relaxed);
        auto_persisted_bytes_.store(0, std::memory_order_relaxed);
        auto_evicted_groups_.store(0, std::memory_order_relaxed);
        auto_evicted_bytes_.store(0, std::memory_order_relaxed);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
        pre_completed_groups_.clear();
        current_bytes_ = 0;
        total_entries_ = 0;
        lru_head_.lru_next = &lru_head_;
        lru_head_.lru_prev = &lru_head_;
        unpersisted_head_.unpersisted_next = &unpersisted_head_;
        unpersisted_head_.unpersisted_prev = &unpersisted_head_;
    }

  private:
    void recompute_watermark_bytes_unlocked() {
        if (capacity_bytes_ > 0) {
            high_watermark_bytes_ = static_cast<size_t>(capacity_bytes_ * high_watermark_);
            low_watermark_bytes_ = static_cast<size_t>(capacity_bytes_ * low_watermark_);
        } else {
            high_watermark_bytes_ = 0;
            low_watermark_bytes_ = 0;
        }
    }

    void ensure_spill_worker_started() {
        std::lock_guard<std::mutex> lk(spill_cv_mutex_);
        if (spill_worker_.joinable() || spill_stop_.load(std::memory_order_acquire)) {
            return;
        }
        spill_worker_ = std::thread([this]() {
            pthread_setname_np(pthread_self(), "IAXL-Spill");
            TaskQueue::bind_cpus();
            setpriority(PRIO_PROCESS, 0, 10);
#ifdef SYS_ioprio_set
            syscall(SYS_ioprio_set, 1, 0, (2 << 13) | 7);
#endif
            spill_worker_loop();
        });
    }

    void maybe_notify_spill() {
        {
            std::lock_guard<std::mutex> lk(spill_cv_mutex_);
            spill_wakeup_requested_ = true;
        }
        spill_cv_.notify_one();
    }

    bool should_run_auto_spill() const {
        if (!auto_spill_enabled_.load(std::memory_order_acquire)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return high_watermark_bytes_ > 0 && current_bytes_ >= high_watermark_bytes_;
    }

    void spill_worker_loop() {
        while (true) {
            {
                std::unique_lock<std::mutex> lk(spill_cv_mutex_);
                spill_cv_.wait(lk, [this]() {
                    return spill_stop_.load(std::memory_order_acquire) || spill_wakeup_requested_ ||
                           should_run_auto_spill();
                });
                if (spill_stop_.load(std::memory_order_acquire)) {
                    break;
                }
                spill_wakeup_requested_ = false;
            }

            if (!should_run_auto_spill()) {
                continue;
            }

            spill_running_.store(true, std::memory_order_release);
            {
                std::lock_guard<std::mutex> op_lk(spill_op_mutex_);
                size_t low_wm_bytes = 0;
                bool evict_only = false;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    low_wm_bytes = low_watermark_bytes_;
                    evict_only = auto_spill_evict_only_.load(std::memory_order_acquire);
                }
                run_spill_loop(low_wm_bytes, /*use_target_bytes=*/true, /*max_groups=*/0,
                               evict_only, /*is_auto_worker=*/true);
            }
            spill_running_.store(false, std::memory_order_release);
        }
    }

    bool is_candidate_eligible_unlocked(const EntryGroup &group, bool allow_incomplete,
                                        double now) const {
        if (group.is_persisting || group.is_unpublishing || group.entries.empty()) {
            return false;
        }
        if (allow_incomplete || !has_complete_tracking_ || group.is_complete) {
            return true;
        }
        // Fallback safety: if a group has not been written for > 50ms, treat it as eligible
        return (now - group.last_access) > 0.05;
    }

    SpillResult run_spill_loop(size_t target_bytes, bool use_target_bytes, size_t max_groups,
                               bool evict_only, bool is_auto_worker) {
        SpillResult result;
        if (!evict_enabled_.load(std::memory_order_acquire)) {
            return result;
        }

        if (is_auto_worker) {
            spill_rounds_.fetch_add(1, std::memory_order_relaxed);
        }

        size_t groups_processed = 0;
        while (!spill_stop_.load(std::memory_order_acquire)) {
            if (is_auto_worker && !auto_spill_enabled_.load(std::memory_order_acquire)) {
                break;
            }
            if (max_groups > 0 && groups_processed >= max_groups) {
                break;
            }

            size_t cur_bytes = 0;
            size_t pace_ms = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                cur_bytes = current_bytes_;
                pace_ms = spill_pace_ms_;
            }
            if (use_target_bytes && cur_bytes <= target_bytes) {
                break;
            }

            if (is_auto_worker && pace_ms > 0 &&
                (active_gets_.load(std::memory_order_relaxed) > 0 || is_deletion_guarded())) {
                std::this_thread::sleep_for(std::chrono::milliseconds(pace_ms));
            }

            bool made_progress = false;
            if (evict_only) {
                size_t step_limit = 1;
                if (max_groups > 0) {
                    step_limit = std::min<size_t>(spill_batch_groups_, max_groups - groups_processed);
                } else {
                    // Estimate how many groups to evict in this batch without overshooting low watermark
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (use_target_bytes && current_bytes_ > target_bytes && !cache_.empty()) {
                        size_t avg_bytes = std::max<size_t>(1, current_bytes_ / cache_.size());
                        size_t excess = current_bytes_ - target_bytes;
                        size_t est = std::max<size_t>(1, (excess + avg_bytes - 1) / avg_bytes);
                        step_limit = std::min<size_t>(spill_batch_groups_, est);
                    } else {
                        step_limit = spill_batch_groups_;
                    }
                }
                made_progress = spill_step_evict_only(step_limit, is_auto_worker, result,
                                                      groups_processed);
            } else {
                made_progress =
                    spill_step_persist_then_evict(is_auto_worker, result, groups_processed);
            }

            if (!made_progress) {
                break;
            }
            result.triggered = true;
            result.rounds++;
        }

        return result;
    }

    bool spill_step_persist_then_evict(bool is_auto_worker, SpillResult &result,
                                       size_t &groups_processed) {
        if (!storage_) {
            return false;
        }

        struct CandidateSnap {
            std::string group_key;
            std::string rep_full_label;
            uint64_t version = 0;
            uint64_t access_seq = 0;
            bool already_persisted = false;
            std::vector<std::pair<std::string, std::vector<char>>> entries;
        };

        CandidateSnap cand;
        bool found = false;

        // Step 1: pick the single oldest LRU group under lock (microsecond critical section)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            double now = now_timestamp();
            for (int pass = 0; pass < 2 && !found; ++pass) {
                bool allow_incomplete = (pass == 1);
                EntryGroup *group = lru_head_.lru_next;
                while (group != &lru_head_) {
                    if (is_candidate_eligible_unlocked(*group, allow_incomplete, now)) {
                        cand.group_key = *group->group_key_ptr;
                        cand.rep_full_label =
                            make_full_label(cand.group_key, group->entries.begin()->first);
                        cand.version = group->version;
                        cand.access_seq = group->access_seq;
                        cand.already_persisted = group->is_persisted;

                        if (cand.already_persisted && !is_deletion_guarded()) {
                            // Fast path: already persisted and no deletion guard active -> evict immediately!
                            size_t freed = remove_unlocked(cand.rep_full_label);
                            evictions_.fetch_add(1, std::memory_order_relaxed);
                            result.evicted_groups++;
                            result.evicted_bytes += freed;
                            result.evicted_labels.push_back(cand.group_key);
                            if (is_auto_worker) {
                                auto_evicted_groups_.fetch_add(1, std::memory_order_relaxed);
                                auto_evicted_bytes_.fetch_add(freed, std::memory_order_relaxed);
                            }
                            groups_processed++;
                            return true;
                        }

                        group->is_persisting = true;
                        if (!cand.already_persisted) {
                            cand.entries.reserve(group->entries.size());
                            for (const auto &[tensor_key, sub] : group->entries) {
                                if (sub.data && sub.size > 0) {
                                    std::string fl = make_full_label(cand.group_key, tensor_key);
                                    cand.entries.emplace_back(
                                        std::move(fl),
                                        std::vector<char>(sub.data, sub.data + sub.size));
                                }
                            }
                        }
                        found = true;
                        break;
                    }
                    group = group->lru_next;
                }
            }
        }

        if (!found) {
            return false;
        }

        // Step 2: perform disk I/O and Record update outside lock
        bool persist_ok = cand.already_persisted;
        size_t written_bytes = 0;
        if (!cand.already_persisted) {
            std::vector<StorageBackend::WriteItem> write_items;
            write_items.reserve(cand.entries.size());
            for (const auto &[fl, buf] : cand.entries) {
                write_items.push_back({fl, buf.data(), buf.size()});
                written_bytes += buf.size();
            }
            auto save_res = storage_->save_batch(write_items);
            persist_ok = !save_res.empty();
            for (bool ok : save_res) {
                if (!ok) {
                    persist_ok = false;
                    break;
                }
            }
            if (persist_ok && record_) {
                record_->mark_persisted({cand.group_key});
            }
        }

        // Wait for any active decompression guard outside lock
        wait_deletion_guard();

        // Step 3: verify version & access_seq under lock, then mark persisted and evict if still cold
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = cache_.find(cand.group_key);
            if (it == cache_.end()) {
                return true;
            }
            EntryGroup &group = it->second;
            group.is_persisting = false;

            if (!cand.already_persisted && persist_ok) {
                if (group.version == cand.version && !group.is_persisted) {
                    group.is_persisted = true;
                    unpersisted_remove(&group);
                }
                result.persisted_groups++;
                result.persisted_bytes += written_bytes;
                result.persisted_labels.push_back(cand.group_key);
                if (is_auto_worker) {
                    auto_persisted_groups_.fetch_add(1, std::memory_order_relaxed);
                    auto_persisted_bytes_.fetch_add(written_bytes, std::memory_order_relaxed);
                }
            }

            groups_processed++;

            // Only evict if the group is persisted, unmodified, AND was NOT accessed by GET
            // while persistence was in flight!
            if (group.is_persisted && group.version == cand.version &&
                group.access_seq == cand.access_seq && !is_deletion_guarded()) {
                size_t freed = remove_unlocked(cand.rep_full_label);
                evictions_.fetch_add(1, std::memory_order_relaxed);
                result.evicted_groups++;
                result.evicted_bytes += freed;
                result.evicted_labels.push_back(cand.group_key);
                if (is_auto_worker) {
                    auto_evicted_groups_.fetch_add(1, std::memory_order_relaxed);
                    auto_evicted_bytes_.fetch_add(freed, std::memory_order_relaxed);
                }
            }
        }

        return true;
    }

    bool spill_step_evict_only(size_t step_limit, bool is_auto_worker, SpillResult &result,
                               size_t &groups_processed) {
        struct EvictItem {
            std::string group_key;
            std::string rep_full_label;
            uint64_t version = 0;
            uint64_t access_seq = 0;
            bool was_persisted = false;
        };

        std::vector<EvictItem> batch;
        size_t grace_ms = 0;

        // Stage 1: select oldest LRU groups under lock and mark them unpublishing
        {
            std::lock_guard<std::mutex> lock(mutex_);
            grace_ms = evict_grace_ms_;
            double now = now_timestamp();
            for (int pass = 0; pass < 2 && batch.empty(); ++pass) {
                bool allow_incomplete = (pass == 1);
                EntryGroup *group = lru_head_.lru_next;
                while (group != &lru_head_ && batch.size() < step_limit) {
                    EntryGroup *next = group->lru_next;
                    if (is_candidate_eligible_unlocked(*group, allow_incomplete, now)) {
                        group->is_unpublishing = true;
                        const std::string &gk = *group->group_key_ptr;
                        const std::string &tk = group->entries.begin()->first;
                        batch.push_back({gk, make_full_label(gk, tk), group->version,
                                         group->access_seq, group->is_persisted});
                    }
                    group = next;
                }
            }
        }

        if (batch.empty()) {
            return false;
        }

        // Stage 2: remove unpersisted keys from Record outside lock first (Unpublish),
        // then wait the grace window so any in-flight GET that already passed Scheduler has()
        // still finds the data in Mem!
        std::vector<std::string> unpersisted_keys;
        for (const auto &item : batch) {
            if (!item.was_persisted) {
                unpersisted_keys.push_back(item.group_key);
            }
        }
        if (record_ && !unpersisted_keys.empty()) {
            record_->remove(unpersisted_keys);
            if (is_auto_worker && grace_ms > 0 && !spill_stop_.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(grace_ms));
            }
        }

        wait_deletion_guard();

        // Stage 3: physically free groups that were not touched during the unpublish window
        std::unordered_map<std::string, std::vector<std::string>> resubmit_by_label;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto &item : batch) {
                auto it = cache_.find(item.group_key);
                if (it == cache_.end())
                    continue;
                EntryGroup &group = it->second;
                group.is_unpublishing = false;
                groups_processed++;

                if (!is_deletion_guarded() && group.version == item.version &&
                    group.access_seq == item.access_seq) {
                    size_t freed = remove_unlocked(item.rep_full_label);
                    evictions_.fetch_add(1, std::memory_order_relaxed);
                    result.evicted_groups++;
                    result.evicted_bytes += freed;
                    result.evicted_labels.push_back(item.group_key);
                    if (is_auto_worker) {
                        auto_evicted_groups_.fetch_add(1, std::memory_order_relaxed);
                        auto_evicted_bytes_.fetch_add(freed, std::memory_order_relaxed);
                    }
                } else if (record_ && !item.was_persisted && !group.is_persisted) {
                    // Accessed by an in-flight GET during the grace window: restore Record entry!
                    auto [label_sv, chunk_sv] = parse_group_key(item.group_key);
                    resubmit_by_label[std::string(label_sv)].emplace_back(chunk_sv);
                }
            }
        }

        if (record_) {
            for (const auto &[label, chunks] : resubmit_by_label) {
                record_->submit(label, chunks);
            }
        }

        return true;
    }

    void put_loaded_one_unlocked(const std::string &full_label, char *data, size_t size,
                                 double now) {
        auto [group_key_sv, tensor_key_sv] = parse_full_label(full_label);
        std::string group_key(group_key_sv);
        std::string tensor_key(tensor_key_sv);

        auto it = cache_.find(group_key);
        if (it != cache_.end()) {
            EntryGroup &group = it->second;
            SubEntry *sub = group.find(tensor_key_sv);
            if (sub && sub->data) {
                // Concurrent thread already populated this sub-entry; keep existing live buffer
                free(data);
            } else {
                SubEntry &new_sub = group.get_or_create(tensor_key);
                new_sub.data = data;
                new_sub.size = size;
                current_bytes_ += size;
                total_entries_++;
            }
            group.last_access = now;
            group.access_seq = ++global_access_seq_;
            lru_remove(&group);
            lru_push_back(&group);
            if (!group.is_persisted && group.unpersisted_next == nullptr &&
                group.unpersisted_prev == nullptr) {
                group.is_persisted = true;
            }
        } else {
            auto [new_it, inserted] =
                cache_.emplace(std::piecewise_construct, std::forward_as_tuple(group_key),
                               std::forward_as_tuple());
            (void)inserted;
            EntryGroup &group = new_it->second;
            group.group_key_ptr = &new_it->first;

            SubEntry &sub = group.get_or_create(tensor_key);
            sub.data = data;
            sub.size = size;
            group.created_at = now;
            group.last_access = now;
            group.version = 1;
            group.access_seq = ++global_access_seq_;
            group.is_persisted = true;
            group.is_complete = true;

            current_bytes_ += size;
            total_entries_++;

            lru_push_back(&group);
        }
    }

    void put_one_unlocked(const std::string &full_label, char *data, size_t size, size_t unzip_size,
                          double now) {
        if (!data) {
            throw std::runtime_error(
                "[Mem] memory allocation failed for '" + full_label + "' (" + std::to_string(size) +
                " bytes). DDR pool usage: " + std::to_string(current_bytes_ / (1024 * 1024)) +
                " MB / " + std::to_string(capacity_bytes_ / (1024 * 1024)) +
                " MB capacity. Increase IAXL_DDR_POOL_SIZE_GB or enable cache eviction.");
        }

        if (unzip_size > 0) {
            total_unzip_bytes_.fetch_add(unzip_size, std::memory_order_relaxed);
            total_zip_bytes_.fetch_add(size, std::memory_order_relaxed);
        }
        auto [group_key_sv, tensor_key_sv] = parse_full_label(full_label);
        std::string group_key(group_key_sv);
        std::string tensor_key(tensor_key_sv);

        auto it = cache_.find(group_key);
        if (it != cache_.end()) {

            EntryGroup &group = it->second;
            SubEntry *sub = group.find(tensor_key_sv);
            if (sub) {
                current_bytes_ -= sub->size;
                if (sub->data)
                    free(sub->data);
                sub->data = data;
                sub->size = size;
            } else {
                SubEntry &new_sub = group.get_or_create(tensor_key);
                new_sub.data = data;
                new_sub.size = size;
                total_entries_++;

                if (group.entries.size() > (size_t)envs.IAXL_CACHE_CACHEGROUP_SIZE) {
                    fprintf(stderr,
                            "[Mem] WARN: EntryGroup '%s' entries (%zu) "
                            "exceeded reserve (%zu), rehash triggered\n",
                            group_key.c_str(), group.entries.size(),
                            (size_t)envs.IAXL_CACHE_CACHEGROUP_SIZE);
                }
            }
            current_bytes_ += size;
            group.last_access = now;
            group.version++;
            group.access_seq = ++global_access_seq_;
            if (pre_completed_groups_.erase(group_key) > 0) {
                group.is_complete = true;
            }

            lru_remove(&group);
            lru_push_back(&group);

            if (group.is_persisted) {
                group.is_persisted = false;
                unpersisted_push_back(&group);
            }
        } else {

            decltype(cache_.begin()) new_it;
            try {
                auto [it2, inserted] =
                    cache_.emplace(std::piecewise_construct, std::forward_as_tuple(group_key),
                                   std::forward_as_tuple());
                new_it = it2;
            } catch (const std::bad_alloc &) {
                free(data);
                throw std::runtime_error(
                    "[Mem] failed to allocate new cache group '" + group_key +
                    "'. DDR pool usage: " + std::to_string(current_bytes_ / (1024 * 1024)) +
                    " MB / " + std::to_string(capacity_bytes_ / (1024 * 1024)) + " MB capacity, " +
                    std::to_string(cache_.size()) +
                    " groups. System memory exhausted. "
                    "Increase IAXL_DDR_POOL_SIZE_GB or enable cache eviction.");
            }
            EntryGroup &group = new_it->second;
            group.group_key_ptr = &new_it->first;

            SubEntry &sub = group.get_or_create(tensor_key);
            sub.data = data;
            sub.size = size;
            group.created_at = now;
            group.last_access = now;
            group.version = 1;
            group.access_seq = ++global_access_seq_;
            group.is_persisted = false;
            if (pre_completed_groups_.erase(group_key) > 0) {
                group.is_complete = true;
            }

            current_bytes_ += size;
            total_entries_++;

            lru_push_back(&group);
            unpersisted_push_back(&group);

            if (cache_.size() > (size_t)envs.IAXL_CACHE_CACHEGROUP_NUM) {
                fprintf(stderr,
                        "[Mem] WARN: cache groups (%zu) "
                        "exceeded reserve (%zu), rehash triggered\n",
                        cache_.size(), (size_t)envs.IAXL_CACHE_CACHEGROUP_NUM);
            }
        }

        puts_.fetch_add(1, std::memory_order_relaxed);

        if (capacity_bytes_ > 0 && current_bytes_ > capacity_bytes_) {
            static std::atomic<uint64_t> warn_count{0};
            if ((warn_count.fetch_add(1) % 10000) == 0) {
                fprintf(stderr, "[Mem] WARNING: Over budget! %zu MB > %zu MB capacity\n",
                        current_bytes_ / (1024 * 1024), capacity_bytes_ / (1024 * 1024));
            }
        }
    }

    std::pair<const char *, size_t> get_one_unlocked(const std::string &full_label, double now) {
        auto [group_key_sv, tensor_key_sv] = parse_full_label(full_label);
        auto it = cache_.find(std::string(group_key_sv));
        if (it != cache_.end()) {
            EntryGroup &group = it->second;
            SubEntry *sub = group.find(tensor_key_sv);
            if (sub && sub->data) {
                group.last_access = now;
                group.access_seq = ++global_access_seq_;
                lru_remove(&group);
                lru_push_back(&group);
                hits_.fetch_add(1, std::memory_order_relaxed);
                return {sub->data, sub->size};
            }
        }
        return {nullptr, 0};
    }

    size_t evict_unlocked(size_t target_bytes,
                          std::vector<std::string> *unpersisted_groups_to_remove = nullptr) {
        size_t evicted = 0;
        EntryGroup *curr = lru_head_.lru_next;
        while (current_bytes_ > target_bytes && curr != &lru_head_) {
            EntryGroup *next = curr->lru_next;
            if (curr->is_persisting || curr->is_unpublishing) {
                curr = next;
                continue;
            }
            size_t bytes = curr->total_bytes();
            std::string group_key = *curr->group_key_ptr;

            if (unpersisted_groups_to_remove && !curr->is_persisted) {
                unpersisted_groups_to_remove->push_back(group_key);
            }

            evicted += bytes;
            current_bytes_ -= bytes;
            total_entries_ -= curr->entry_count();

            lru_remove(curr);
            unpersisted_remove(curr);

            cache_.erase(group_key);
            evictions_.fetch_add(1, std::memory_order_relaxed);
            curr = next;
        }
        return evicted;
    }

    void lru_remove(EntryGroup *group) {
        if (group->lru_prev) {
            group->lru_prev->lru_next = group->lru_next;
        }
        if (group->lru_next) {
            group->lru_next->lru_prev = group->lru_prev;
        }
        group->lru_prev = nullptr;
        group->lru_next = nullptr;
    }

    void lru_push_back(EntryGroup *group) {
        group->lru_prev = lru_head_.lru_prev;
        group->lru_next = &lru_head_;
        lru_head_.lru_prev->lru_next = group;
        lru_head_.lru_prev = group;
    }

    void unpersisted_remove(EntryGroup *group) {
        if (group->unpersisted_prev) {
            group->unpersisted_prev->unpersisted_next = group->unpersisted_next;
        }
        if (group->unpersisted_next) {
            group->unpersisted_next->unpersisted_prev = group->unpersisted_prev;
        }
        group->unpersisted_prev = nullptr;
        group->unpersisted_next = nullptr;
    }

    void unpersisted_push_back(EntryGroup *group) {
        group->unpersisted_prev = unpersisted_head_.unpersisted_prev;
        group->unpersisted_next = &unpersisted_head_;
        unpersisted_head_.unpersisted_prev->unpersisted_next = group;
        unpersisted_head_.unpersisted_prev = group;
    }

    std::unordered_map<std::string, EntryGroup> cache_;
    std::unordered_set<std::string> pre_completed_groups_;
    bool has_complete_tracking_ = false;
    uint64_t global_access_seq_ = 0;

    EntryGroup lru_head_;
    EntryGroup unpersisted_head_;

    size_t capacity_bytes_;
    size_t current_bytes_;
    size_t total_entries_;

    std::atomic<bool> evict_enabled_;

    mutable std::mutex mutex_;

    std::atomic<int> deletion_guard_count_{0};
    std::mutex deletion_cv_mutex_;
    std::condition_variable deletion_cv_;
    std::atomic<int> active_gets_{0};

    std::atomic<bool> auto_spill_enabled_{false};
    std::atomic<bool> auto_spill_evict_only_{false};
    std::atomic<bool> spill_running_{false};
    std::atomic<bool> spill_stop_{false};
    bool spill_wakeup_requested_ = false;
    double high_watermark_ = 0.80;
    double low_watermark_ = 0.60;
    size_t high_watermark_bytes_ = 0;
    size_t low_watermark_bytes_ = 0;
    size_t spill_batch_groups_ = 16;
    size_t spill_pace_ms_ = 2;
    size_t evict_grace_ms_ = 50;

    std::thread spill_worker_;
    std::mutex spill_cv_mutex_;
    std::condition_variable spill_cv_;
    std::mutex spill_op_mutex_;

    std::atomic<uint64_t> spill_rounds_{0};
    std::atomic<uint64_t> auto_persisted_groups_{0};
    std::atomic<uint64_t> auto_persisted_bytes_{0};
    std::atomic<uint64_t> auto_evicted_groups_{0};
    std::atomic<uint64_t> auto_evicted_bytes_{0};

    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> hits_in_storage_{0};
    std::atomic<uint64_t> puts_{0};
    std::atomic<uint64_t> evictions_{0};

    Storage *storage_ = nullptr;
    Record *record_ = nullptr;

    std::atomic<uint64_t> total_unzip_bytes_{0};
    std::atomic<uint64_t> total_zip_bytes_{0};
};

Mem::Mem(size_t capacity_bytes, Storage *storage, Record *record)
    : impl_(std::make_unique<Impl>(capacity_bytes, storage, record)) {}
Mem::~Mem() = default;

void Mem::put(const std::vector<std::string> &keys, std::vector<char *> &&data_ptrs,
              const std::vector<size_t> &sizes, const std::vector<size_t> &unzip_sizes) {
    impl_->put(keys, std::move(data_ptrs), sizes, unzip_sizes);
}

std::vector<std::pair<const char *, size_t>> Mem::get(const std::vector<std::string> &keys,
                                                      bool allow_using_omp,
                                                      bool caller_holds_deletion_guard) {
    return impl_->get(keys, allow_using_omp, caller_holds_deletion_guard);
}

std::vector<std::pair<const char *, size_t>> Mem::get_unlocked(const std::vector<std::string> &keys,
                                                               bool allow_using_omp) {
    return impl_->get_unlocked(keys, allow_using_omp);
}

std::vector<bool> Mem::has(const std::vector<std::string> &keys) { return impl_->has(keys); }

size_t Mem::remove(const std::string &key) { return impl_->remove(key); }

std::vector<std::tuple<std::string, size_t>> Mem::persist_groups(size_t max_groups) {
    return impl_->persist_groups(max_groups);
}

std::vector<std::tuple<std::string, size_t>> Mem::evict_groups(size_t max_groups) {
    return impl_->evict_groups(max_groups);
}

SpillResult Mem::spill_groups(size_t max_groups, int evict_only_override) {
    return impl_->spill_groups(max_groups, evict_only_override);
}

SpillResult Mem::spill_to_size(size_t target_bytes, size_t max_groups, int evict_only_override) {
    return impl_->spill_to_size(target_bytes, max_groups, evict_only_override);
}

void Mem::configure_auto_spill(bool enabled, bool evict_only, double high_watermark,
                               double low_watermark, size_t batch_groups, size_t pace_ms,
                               size_t evict_grace_ms) {
    impl_->configure_auto_spill(enabled, evict_only, high_watermark, low_watermark, batch_groups,
                                pace_ms, evict_grace_ms);
}

AutoSpillStatus Mem::auto_spill_status() const { return impl_->auto_spill_status(); }

void Mem::mark_groups_complete(const std::string &label,
                               const std::vector<std::string> &chunk_ids) {
    impl_->mark_groups_complete(label, chunk_ids);
}

void Mem::stop_background_workers() { impl_->stop_background_workers(); }

size_t Mem::evict_to_size(size_t target_bytes) { return impl_->evict_to_size(target_bytes); }
size_t Mem::force_evict_to_size(size_t target_bytes) {
    return impl_->force_evict_to_size(target_bytes);
}

void Mem::set_evict_enabled(bool enabled) { impl_->set_evict_enabled(enabled); }
bool Mem::is_evict_enabled() const { return impl_->is_evict_enabled(); }

void Mem::lock() { impl_->lock(); }
void Mem::unlock() { impl_->unlock(); }

void Mem::acquire_deletion_guard() { impl_->acquire_deletion_guard(); }
void Mem::release_deletion_guard() { impl_->release_deletion_guard(); }
bool Mem::is_deletion_guarded() const { return impl_->is_deletion_guarded(); }

std::vector<std::tuple<std::string, const char *, size_t>>
Mem::get_unpersisted_unlocked(size_t max_count) {
    return impl_->get_unpersisted_unlocked(max_count);
}

std::vector<std::tuple<std::string, const char *, size_t>>
Mem::get_lru_oldest_unlocked(size_t max_count) {
    return impl_->get_lru_oldest_unlocked(max_count);
}

void Mem::mark_persisted(const std::vector<std::string> &keys) { impl_->mark_persisted(keys); }

size_t Mem::size() const { return impl_->size(); }
size_t Mem::group_count() const { return impl_->group_count(); }
size_t Mem::current_bytes() const { return impl_->current_bytes(); }
size_t Mem::capacity_bytes() const { return impl_->capacity_bytes(); }
void Mem::set_capacity_bytes(size_t capacity) { impl_->set_capacity_bytes(capacity); }

uint64_t Mem::hits() const { return impl_->hits(); }
uint64_t Mem::hits_in_storage() const { return impl_->hits_in_storage(); }
uint64_t Mem::misses() const { return impl_->misses(); }
uint64_t Mem::puts() const { return impl_->puts(); }
uint64_t Mem::evictions() const { return impl_->evictions(); }
uint64_t Mem::total_unzip_bytes() const { return impl_->total_unzip_bytes(); }
uint64_t Mem::total_zip_bytes() const { return impl_->total_zip_bytes(); }
double Mem::compression_ratio() const { return impl_->compression_ratio(); }
size_t Mem::unpersisted_count() const { return impl_->unpersisted_count(); }

void Mem::reset_stats() { impl_->reset_stats(); }
void Mem::clear() { impl_->clear(); }

} // namespace kv_pool
