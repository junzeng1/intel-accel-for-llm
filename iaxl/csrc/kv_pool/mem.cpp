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
    bool prefetched_from_storage = false;

    ~SubEntry() {
        if (data) {
            free(data);
            data = nullptr;
        }
    }

    SubEntry(const SubEntry &) = delete;
    SubEntry &operator=(const SubEntry &) = delete;

    SubEntry() = default;

    SubEntry(SubEntry &&other) noexcept
        : data(other.data), size(other.size),
          prefetched_from_storage(other.prefetched_from_storage) {
        other.data = nullptr;
        other.size = 0;
        other.prefetched_from_storage = false;
    }

    SubEntry &operator=(SubEntry &&other) noexcept {
        if (this != &other) {
            if (data)
                free(data);
            data = other.data;
            size = other.size;
            prefetched_from_storage = other.prefetched_from_storage;
            other.data = nullptr;
            other.size = 0;
            other.prefetched_from_storage = false;
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
    uint32_t active_readers = 0;

    uint64_t version = 0;
    uint64_t access_seq = 0;

    const std::string *group_key_ptr = nullptr;

    double created_at = 0.0;
    double last_access = 0.0;

    std::unordered_map<std::string, SubEntry> entries;
    std::vector<char *> retired_buffers;

    void flush_retired_buffers() {
        if (active_readers == 0 && !is_persisting && !retired_buffers.empty()) {
            for (char *p : retired_buffers) {
                if (p)
                    free(p);
            }
            retired_buffers.clear();
        }
    }

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

    ~EntryGroup() {
        for (char *p : retired_buffers) {
            if (p)
                free(p);
        }
        retired_buffers.clear();
    }

    EntryGroup() { entries.reserve((size_t)envs.IAXL_CACHE_CACHEGROUP_SIZE); }

    EntryGroup(const EntryGroup &) = delete;
    EntryGroup &operator=(const EntryGroup &) = delete;

    EntryGroup(EntryGroup &&other) noexcept
        : lru_prev(other.lru_prev), lru_next(other.lru_next),
          unpersisted_prev(other.unpersisted_prev), unpersisted_next(other.unpersisted_next),
          is_persisted(other.is_persisted), is_persisting(other.is_persisting),
          is_unpublishing(other.is_unpublishing), is_complete(other.is_complete),
          active_readers(other.active_readers), version(other.version),
          access_seq(other.access_seq), group_key_ptr(other.group_key_ptr),
          created_at(other.created_at), last_access(other.last_access),
          entries(std::move(other.entries)), retired_buffers(std::move(other.retired_buffers)) {
        other.lru_prev = nullptr;
        other.lru_next = nullptr;
        other.unpersisted_prev = nullptr;
        other.unpersisted_next = nullptr;
        other.active_readers = 0;
        other.group_key_ptr = nullptr;
    }

    EntryGroup &operator=(EntryGroup &&other) noexcept {
        if (this != &other) {
            for (char *p : retired_buffers) {
                if (p)
                    free(p);
            }
            lru_prev = other.lru_prev;
            lru_next = other.lru_next;
            unpersisted_prev = other.unpersisted_prev;
            unpersisted_next = other.unpersisted_next;
            is_persisted = other.is_persisted;
            is_persisting = other.is_persisting;
            is_unpublishing = other.is_unpublishing;
            is_complete = other.is_complete;
            active_readers = other.active_readers;
            version = other.version;
            access_seq = other.access_seq;
            group_key_ptr = other.group_key_ptr;
            created_at = other.created_at;
            last_access = other.last_access;
            entries = std::move(other.entries);
            retired_buffers = std::move(other.retired_buffers);
            other.lru_prev = nullptr;
            other.lru_next = nullptr;
            other.unpersisted_prev = nullptr;
            other.unpersisted_next = nullptr;
            other.active_readers = 0;
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
        bool should_notify = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            has_complete_tracking_ = true;
            for (const auto &gk : group_keys) {
                auto it = cache_.find(gk);
                if (it != cache_.end()) {
                    EntryGroup &group = it->second;
                    if (expected_group_size_ == 0 || label != "kv" ||
                        group.entries.size() >= expected_group_size_) {
                        group.is_complete = true;
                    } else {
                        pre_completed_groups_.insert(gk);
                    }
                } else {
                    pre_completed_groups_.insert(gk);
                }
            }
            should_notify = should_notify_spill_unlocked();
        }
        if (should_notify) {
            maybe_notify_spill();
        }
    }

    void set_expected_group_size(size_t expected_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        expected_group_size_ = expected_size;
    }

    size_t expected_group_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return expected_group_size_;
    }

    void release_group_guards(const std::vector<std::string> &keys) {
        if (keys.empty())
            return;
        int released = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto &key : keys) {
                auto [group_key_sv, _] = parse_full_label(key);
                auto it = cache_.find(std::string(group_key_sv));
                if (it != cache_.end()) {
                    EntryGroup &group = it->second;
                    if (group.active_readers > 0) {
                        group.active_readers--;
                        group.flush_retired_buffers();
                        released++;
                    }
                }
            }
        }
        if (released > 0) {
            int prev = active_group_readers_total_.fetch_sub(released, std::memory_order_acq_rel);
            if (prev <= released) {
                std::lock_guard<std::mutex> lk(deletion_cv_mutex_);
                deletion_cv_.notify_all();
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

    void wait_all_readers() {
        if (deletion_guard_count_.load(std::memory_order_acquire) == 0 &&
            active_group_readers_total_.load(std::memory_order_acquire) == 0)
            return;
        std::unique_lock<std::mutex> lk(deletion_cv_mutex_);
        deletion_cv_.wait(lk, [this] {
            return deletion_guard_count_.load(std::memory_order_acquire) == 0 &&
                   active_group_readers_total_.load(std::memory_order_acquire) == 0;
        });
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
            should_notify = should_notify_spill_unlocked();
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
        if (n == 0) {
            return results;
        }
        double now = now_timestamp();
        const bool use_group_pins = caller_holds_deletion_guard && !is_deletion_guarded();

        // Phase 1: fast in-memory lookup under lock
        std::vector<size_t> miss_indices;
        std::vector<std::string> phase1_pinned_groups;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            miss_indices.reserve(n);
            std::vector<EntryGroup *> hit_groups;
            hit_groups.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                EntryGroup *g = nullptr;
                results[i] = get_one_unlocked(keys[i], now, &g);
                if (results[i].first == nullptr) {
                    miss_indices.push_back(i);
                } else if (g != nullptr) {
                    hit_groups.push_back(g);
                }
            }

            if (miss_indices.empty()) {
                if (use_group_pins && !hit_groups.empty()) {
                    for (EntryGroup *g : hit_groups) {
                        g->active_readers++;
                    }
                    active_group_readers_total_.fetch_add(static_cast<int>(hit_groups.size()),
                                                          std::memory_order_acq_rel);
                }
                return results;
            }

            // Temporarily pin Phase-1 hits so they cannot be evicted while Phase 2 reads from disk
            if (!hit_groups.empty()) {
                phase1_pinned_groups.reserve(hit_groups.size());
                for (EntryGroup *g : hit_groups) {
                    g->active_readers++;
                    phase1_pinned_groups.push_back(*g->group_key_ptr);
                }
                active_group_readers_total_.fetch_add(static_cast<int>(hit_groups.size()),
                                                      std::memory_order_acq_rel);
            }
        }

        auto release_phase1_pins = [this, &phase1_pinned_groups]() {
            if (phase1_pinned_groups.empty())
                return;
            int released = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                for (const auto &gk : phase1_pinned_groups) {
                    auto it = cache_.find(gk);
                    if (it != cache_.end() && it->second.active_readers > 0) {
                        it->second.active_readers--;
                        it->second.flush_retired_buffers();
                        released++;
                    }
                }
            }
            phase1_pinned_groups.clear();
            if (released > 0) {
                if (active_group_readers_total_.fetch_sub(released, std::memory_order_acq_rel) <=
                    released) {
                    std::lock_guard<std::mutex> lk(deletion_cv_mutex_);
                    deletion_cv_.notify_all();
                }
            }
        };

        if (!storage_ || !record_) {
            release_phase1_pins();
            misses_.fetch_add(miss_indices.size(), std::memory_order_relaxed);
            return results;
        }

        // If caller explicitly held the legacy global deletion guard, release it during slow disk I/O
        const bool release_global_guard = caller_holds_deletion_guard && is_deletion_guarded();
        if (release_global_guard) {
            release_deletion_guard();
        }

        struct GuardRestorer {
            Impl *self;
            bool active;
            ~GuardRestorer() {
                if (active)
                    self->acquire_deletion_guard();
            }
        } restorer{this, release_global_guard};

        // Phase 2: lock-free deduplicated Record check + Storage whole-group batch load
        std::vector<std::string> unique_miss_groups;
        unique_miss_groups.reserve(miss_indices.size());
        std::unordered_map<std::string, size_t> group_to_unique_idx;
        group_to_unique_idx.reserve(miss_indices.size());
        std::vector<std::vector<size_t>> group_miss_positions;
        group_miss_positions.reserve(miss_indices.size());

        for (size_t pos = 0; pos < miss_indices.size(); ++pos) {
            size_t key_idx = miss_indices[pos];
            auto [group_key_sv, _] = parse_full_label(keys[key_idx]);
            std::string gk(group_key_sv);
            auto [it, inserted] = group_to_unique_idx.emplace(gk, unique_miss_groups.size());
            if (inserted) {
                unique_miss_groups.push_back(std::move(gk));
                group_miss_positions.emplace_back();
            }
            group_miss_positions[it->second].push_back(pos);
        }

        std::vector<bool> group_persisted = record_->is_persisted(unique_miss_groups);
        std::vector<std::string> persisted_group_keys;
        std::vector<size_t> persisted_unique_indices;
        persisted_group_keys.reserve(unique_miss_groups.size());
        persisted_unique_indices.reserve(unique_miss_groups.size());

        size_t unpersisted_miss_count = 0;
        for (size_t u = 0; u < unique_miss_groups.size(); ++u) {
            if (group_persisted[u]) {
                persisted_group_keys.push_back(unique_miss_groups[u]);
                persisted_unique_indices.push_back(u);
            } else {
                unpersisted_miss_count += group_miss_positions[u].size();
            }
        }

        if (unpersisted_miss_count > 0) {
            misses_.fetch_add(unpersisted_miss_count, std::memory_order_relaxed);
        }

        std::vector<StorageBackend::LoadedGroup> loaded_groups;
        if (!persisted_group_keys.empty()) {
            loaded_groups = storage_->load_groups_batch(persisted_group_keys, allow_using_omp);
        }

        // Legacy fallback for any persisted group stored as individual per-tensor .bin files
        std::vector<size_t> fallback_key_indices;
        std::vector<std::string> fallback_load_keys;
        for (size_t k = 0; k < persisted_group_keys.size(); ++k) {
            if (loaded_groups[k].empty()) {
                size_t u = persisted_unique_indices[k];
                for (size_t pos : group_miss_positions[u]) {
                    size_t key_idx = miss_indices[pos];
                    fallback_key_indices.push_back(key_idx);
                    fallback_load_keys.push_back(keys[key_idx]);
                }
            }
        }

        std::vector<std::pair<char *, size_t>> fallback_loaded_buffers;
        if (!fallback_load_keys.empty()) {
            fallback_loaded_buffers = storage_->load_batch(fallback_load_keys, allow_using_omp);
        }

        // Re-acquire legacy global deletion guard if it was held by caller
        if (restorer.active) {
            acquire_deletion_guard();
            restorer.active = false;
        }

        // Phase 3: insert loaded group entries (populating all layers of each group) and resolve pointers
        bool should_notify = false;
        int new_pins = 0;
        int released_phase1_pins = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto &gk : phase1_pinned_groups) {
                auto it = cache_.find(gk);
                if (it != cache_.end() && it->second.active_readers > 0) {
                    it->second.active_readers--;
                    it->second.flush_retired_buffers();
                    released_phase1_pins++;
                }
            }
            phase1_pinned_groups.clear();

            for (size_t k = 0; k < persisted_group_keys.size(); ++k) {
                if (!loaded_groups[k].empty()) {
                    size_t u = persisted_unique_indices[k];
                    std::unordered_set<std::string> requested_tks;
                    requested_tks.reserve(group_miss_positions[u].size());
                    for (size_t pos : group_miss_positions[u]) {
                        auto [_, tk_sv] = parse_full_label(keys[miss_indices[pos]]);
                        requested_tks.emplace(tk_sv);
                    }
                    put_loaded_group_unlocked(persisted_group_keys[k], std::move(loaded_groups[k]),
                                              requested_tks, now);
                }
            }

            for (size_t i = 0; i < fallback_key_indices.size(); ++i) {
                size_t key_index = fallback_key_indices[i];
                char *buf = fallback_loaded_buffers[i].first;
                size_t sz = fallback_loaded_buffers[i].second;
                if (buf) {
                    put_loaded_one_unlocked(keys[key_index], buf, sz, now,
                                            /*prefetched=*/false);
                    hits_in_storage_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    misses_.fetch_add(1, std::memory_order_relaxed);
                }
            }

            // Resolve all pointers and pin groups if requested
            std::vector<EntryGroup *> resolved_groups;
            if (use_group_pins) {
                resolved_groups.reserve(n);
            }
            bool all_resolved = true;
            for (size_t i = 0; i < n; ++i) {
                auto [group_key_sv, tensor_key_sv] = parse_full_label(keys[i]);
                auto it = cache_.find(std::string(group_key_sv));
                if (it != cache_.end()) {
                    SubEntry *sub = it->second.find(tensor_key_sv);
                    if (sub && sub->data) {
                        results[i] = {sub->data, sub->size};
                        if (use_group_pins) {
                            resolved_groups.push_back(&it->second);
                        }
                        continue;
                    }
                }
                results[i] = {nullptr, 0};
                all_resolved = false;
            }

            if (use_group_pins && all_resolved) {
                for (EntryGroup *g : resolved_groups) {
                    g->active_readers++;
                }
                new_pins = static_cast<int>(resolved_groups.size());
            }

            should_notify = should_notify_spill_unlocked();
        }

        if (new_pins != released_phase1_pins) {
            int delta = new_pins - released_phase1_pins;
            int prev = active_group_readers_total_.fetch_add(delta, std::memory_order_acq_rel);
            if (delta < 0 && prev + delta <= 0) {
                std::lock_guard<std::mutex> lk(deletion_cv_mutex_);
                deletion_cv_.notify_all();
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
            uint64_t version = 0;
            size_t total_bytes = 0;
            std::vector<StorageBackend::GroupEntryItem> entries;
        };

        // Phase 1: select unpersisted groups under lock using zero-copy pointers
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
                            gsnap.entries.push_back({tensor_key, sub.data, sub.size});
                            gsnap.total_bytes += sub.size;
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

        // Phase 2: write aggregated group files to Storage outside lock
        std::vector<StorageBackend::GroupWriteItem> write_groups;
        write_groups.reserve(group_snaps.size());
        for (const auto &gsnap : group_snaps) {
            write_groups.push_back({gsnap.group_key, gsnap.entries});
        }
        std::vector<bool> save_results = storage_->save_groups_batch(write_groups, /*allow_parallel=*/true);

        // Phase 3a: verify version under lock BEFORE marking persisted in Record
        std::vector<std::string> valid_persisted_keys;
        std::vector<std::string> stale_persisted_keys;
        valid_persisted_keys.reserve(group_snaps.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t i = 0; i < group_snaps.size(); ++i) {
                if (!save_results[i])
                    continue;
                auto it = cache_.find(group_snaps[i].group_key);
                if (it != cache_.end() && it->second.version == group_snaps[i].version) {
                    valid_persisted_keys.push_back(group_snaps[i].group_key);
                } else {
                    stale_persisted_keys.push_back(group_snaps[i].group_key);
                }
            }
        }

        // Phase 3b: commit Record metadata for unmodified groups only
        if (record_ && !valid_persisted_keys.empty()) {
            record_->mark_persisted(valid_persisted_keys);
        }
        for (const auto &stale_key : stale_persisted_keys) {
            storage_->remove_group(stale_key);
        }

        // Phase 3c: finalize in-memory state under lock
        std::unordered_set<std::string> valid_set(valid_persisted_keys.begin(),
                                                  valid_persisted_keys.end());
        std::vector<std::tuple<std::string, size_t>> persisted_groups;
        std::vector<std::string> late_stale_keys;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto &gsnap : group_snaps) {
                auto it = cache_.find(gsnap.group_key);
                if (it == cache_.end())
                    continue;
                EntryGroup &group = it->second;
                group.is_persisting = false;
                group.flush_retired_buffers();
                if (valid_set.count(gsnap.group_key) > 0) {
                    if (group.version == gsnap.version) {
                        if (!group.is_persisted) {
                            group.is_persisted = true;
                            unpersisted_remove(&group);
                        }
                        persisted_groups.emplace_back(gsnap.group_key, gsnap.total_bytes);
                    } else {
                        late_stale_keys.push_back(gsnap.group_key);
                    }
                }
            }
        }
        for (const auto &stale_key : late_stale_keys) {
            storage_->remove_group(stale_key);
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
                if (!group->is_persisting && !group->is_unpublishing &&
                    group->active_readers == 0 && !group->entries.empty()) {
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
                if (!is_deletion_guarded() && group.active_readers == 0 &&
                    group.version == c.version && group.access_seq == c.access_seq) {
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

        if (!storage_ || !record_) {
            misses_.fetch_add(miss_indices.size(), std::memory_order_relaxed);
            return results;
        }

        std::vector<std::string> unique_miss_groups;
        unique_miss_groups.reserve(miss_indices.size());
        std::unordered_map<std::string, size_t> group_to_unique_idx;
        group_to_unique_idx.reserve(miss_indices.size());
        std::vector<std::vector<size_t>> group_miss_positions;
        group_miss_positions.reserve(miss_indices.size());

        for (size_t pos = 0; pos < miss_indices.size(); ++pos) {
            size_t key_idx = miss_indices[pos];
            auto [group_key_sv, _] = parse_full_label(keys[key_idx]);
            std::string gk(group_key_sv);
            auto [it, inserted] = group_to_unique_idx.emplace(gk, unique_miss_groups.size());
            if (inserted) {
                unique_miss_groups.push_back(std::move(gk));
                group_miss_positions.emplace_back();
            }
            group_miss_positions[it->second].push_back(pos);
        }

        std::vector<bool> group_persisted = record_->is_persisted(unique_miss_groups);
        std::vector<std::string> persisted_group_keys;
        std::vector<size_t> persisted_unique_indices;
        size_t unpersisted_miss_count = 0;
        for (size_t u = 0; u < unique_miss_groups.size(); ++u) {
            if (group_persisted[u]) {
                persisted_group_keys.push_back(unique_miss_groups[u]);
                persisted_unique_indices.push_back(u);
            } else {
                unpersisted_miss_count += group_miss_positions[u].size();
            }
        }
        if (unpersisted_miss_count > 0) {
            misses_.fetch_add(unpersisted_miss_count, std::memory_order_relaxed);
        }
        if (persisted_group_keys.empty()) {
            return results;
        }

        auto loaded_groups = storage_->load_groups_batch(persisted_group_keys, allow_using_omp);

        std::vector<size_t> fallback_key_indices;
        std::vector<std::string> fallback_load_keys;
        for (size_t k = 0; k < persisted_group_keys.size(); ++k) {
            if (!loaded_groups[k].empty()) {
                size_t u = persisted_unique_indices[k];
                std::unordered_set<std::string> requested_tks;
                for (size_t pos : group_miss_positions[u]) {
                    auto [_, tk_sv] = parse_full_label(keys[miss_indices[pos]]);
                    requested_tks.emplace(tk_sv);
                }
                put_loaded_group_unlocked(persisted_group_keys[k], std::move(loaded_groups[k]),
                                          requested_tks, now);
            } else {
                size_t u = persisted_unique_indices[k];
                for (size_t pos : group_miss_positions[u]) {
                    size_t key_idx = miss_indices[pos];
                    fallback_key_indices.push_back(key_idx);
                    fallback_load_keys.push_back(keys[key_idx]);
                }
            }
        }

        if (!fallback_load_keys.empty()) {
            auto fallback_buffers = storage_->load_batch(fallback_load_keys, allow_using_omp);
            for (size_t i = 0; i < fallback_key_indices.size(); ++i) {
                size_t key_index = fallback_key_indices[i];
                if (fallback_buffers[i].first) {
                    put_loaded_one_unlocked(keys[key_index], fallback_buffers[i].first,
                                            fallback_buffers[i].second, now, /*prefetched=*/false);
                    hits_in_storage_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    misses_.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }

        for (size_t idx : miss_indices) {
            auto [group_key_sv, tensor_key_sv] = parse_full_label(keys[idx]);
            auto it = cache_.find(std::string(group_key_sv));
            if (it != cache_.end()) {
                SubEntry *sub = it->second.find(tensor_key_sv);
                if (sub && sub->data) {
                    results[idx] = {sub->data, sub->size};
                }
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
        wait_all_readers();
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
        wait_all_readers();
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

    bool is_group_complete_unlocked(const EntryGroup &group) const {
        if (group.entries.empty()) {
            return false;
        }
        if (expected_group_size_ > 0 && group.group_key_ptr != nullptr &&
            parse_group_key(*group.group_key_ptr).first == "kv") {
            return group.entries.size() >= expected_group_size_;
        }
        if (has_complete_tracking_) {
            return group.is_complete;
        }
        return true;
    }

    bool has_eligible_unpersisted_unlocked(double now) const {
        const EntryGroup *group = unpersisted_head_.unpersisted_next;
        while (group != &unpersisted_head_) {
            if (!group->is_persisting && !group->is_unpublishing && !group->entries.empty()) {
                if (is_group_complete_unlocked(*group)) {
                    return true;
                }
                if (expected_group_size_ == 0 && (now - group->last_access) > 0.05) {
                    return true;
                }
            }
            group = group->unpersisted_next;
        }
        return false;
    }

    bool should_notify_spill_unlocked() const {
        if (!auto_spill_enabled_.load(std::memory_order_relaxed) ||
            spill_running_.load(std::memory_order_relaxed)) {
            return false;
        }
        if (high_watermark_bytes_ > 0 && current_bytes_ >= high_watermark_bytes_) {
            return true;
        }
        if (!auto_spill_evict_only_.load(std::memory_order_relaxed) && storage_ != nullptr &&
            low_watermark_bytes_ > 0 && current_bytes_ >= low_watermark_bytes_) {
            return has_eligible_unpersisted_unlocked(now_timestamp());
        }
        return false;
    }

    bool should_run_auto_spill() const {
        if (!auto_spill_enabled_.load(std::memory_order_acquire)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (high_watermark_bytes_ > 0 && current_bytes_ >= high_watermark_bytes_) {
            return true;
        }
        if (!auto_spill_evict_only_.load(std::memory_order_acquire) && storage_ != nullptr &&
            low_watermark_bytes_ > 0 && current_bytes_ >= low_watermark_bytes_) {
            return has_eligible_unpersisted_unlocked(now_timestamp());
        }
        return false;
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
                size_t high_wm_bytes = 0;
                size_t low_wm_bytes = 0;
                size_t cur_bytes = 0;
                bool evict_only = false;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    high_wm_bytes = high_watermark_bytes_;
                    low_wm_bytes = low_watermark_bytes_;
                    cur_bytes = current_bytes_;
                    evict_only = auto_spill_evict_only_.load(std::memory_order_acquire);
                }
                if (high_wm_bytes > 0 && cur_bytes >= high_wm_bytes) {
                    run_spill_loop(low_wm_bytes, /*use_target_bytes=*/true, /*max_groups=*/0,
                                   evict_only, /*is_auto_worker=*/true);
                }
                if (!evict_only && storage_ != nullptr && low_wm_bytes > 0) {
                    run_pre_persist_loop(low_wm_bytes, high_wm_bytes);
                }
            }
            spill_running_.store(false, std::memory_order_release);
        }
    }

    bool is_candidate_eligible_unlocked(const EntryGroup &group, bool allow_incomplete, double now,
                                        bool for_persist) const {
        if (group.is_persisting || group.is_unpublishing || group.entries.empty()) {
            return false;
        }
        if (!for_persist && group.active_readers > 0) {
            return false;
        }
        if (is_group_complete_unlocked(group)) {
            return true;
        }
        if (for_persist && expected_group_size_ > 0 && group.group_key_ptr != nullptr &&
            parse_group_key(*group.group_key_ptr).first == "kv") {
            // Never persist a half-written KV group unless it has been stalled for > 5s
            return allow_incomplete && (now - group.last_access) > 5.0;
        }
        if (allow_incomplete) {
            return true;
        }
        // Fallback safety when expected_group_size_ is not configured
        return expected_group_size_ == 0 && (now - group.last_access) > 0.05;
    }

    void run_pre_persist_loop(size_t low_wm_bytes, size_t high_wm_bytes) {
        while (!spill_stop_.load(std::memory_order_acquire) &&
               auto_spill_enabled_.load(std::memory_order_acquire) &&
               !auto_spill_evict_only_.load(std::memory_order_acquire)) {
            size_t cur_bytes = 0;
            size_t pace_ms = 0;
            size_t batch_limit = 16;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                cur_bytes = current_bytes_;
                pace_ms = spill_pace_ms_;
                batch_limit = spill_batch_groups_;
            }
            if (cur_bytes < low_wm_bytes) {
                break;
            }
            if (high_wm_bytes > 0 && cur_bytes >= high_wm_bytes) {
                run_spill_loop(low_wm_bytes, /*use_target_bytes=*/true, /*max_groups=*/0,
                               /*evict_only=*/false, /*is_auto_worker=*/true);
                continue;
            }

            if (pace_ms > 0 && (active_gets_.load(std::memory_order_relaxed) > 0 ||
                                active_group_readers_total_.load(std::memory_order_relaxed) > 0 ||
                                is_deletion_guarded())) {
                std::this_thread::sleep_for(std::chrono::milliseconds(pace_ms));
            }

            if (!pre_persist_batch_step(batch_limit)) {
                break;
            }
        }
    }

    bool pre_persist_batch_step(size_t batch_limit) {
        if (!storage_ || batch_limit == 0) {
            return false;
        }

        struct PrePersistSnap {
            std::string group_key;
            uint64_t version = 0;
            size_t written_bytes = 0;
            std::vector<StorageBackend::GroupEntryItem> entries;
        };

        std::vector<PrePersistSnap> batch;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            double now = now_timestamp();
            EntryGroup *group = unpersisted_head_.unpersisted_next;
            while (group != &unpersisted_head_ && batch.size() < batch_limit) {
                EntryGroup *next = group->unpersisted_next;
                if (is_candidate_eligible_unlocked(*group, /*allow_incomplete=*/false, now,
                                                   /*for_persist=*/true)) {
                    group->is_persisting = true;
                    PrePersistSnap snap;
                    snap.group_key = *group->group_key_ptr;
                    snap.version = group->version;
                    snap.entries.reserve(group->entries.size());
                    for (const auto &[tensor_key, sub] : group->entries) {
                        if (sub.data && sub.size > 0) {
                            snap.entries.push_back({tensor_key, sub.data, sub.size});
                            snap.written_bytes += sub.size;
                        }
                    }
                    batch.push_back(std::move(snap));
                }
                group = next;
            }
        }

        if (batch.empty()) {
            return false;
        }

        std::vector<StorageBackend::GroupWriteItem> write_groups;
        write_groups.reserve(batch.size());
        for (const auto &snap : batch) {
            write_groups.push_back({snap.group_key, snap.entries});
        }
        std::vector<bool> save_res =
            storage_->save_groups_batch(write_groups, /*allow_parallel=*/false);

        std::vector<std::string> valid_keys;
        std::vector<std::string> stale_keys;
        valid_keys.reserve(batch.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t i = 0; i < batch.size(); ++i) {
                if (!save_res[i])
                    continue;
                auto it = cache_.find(batch[i].group_key);
                if (it != cache_.end() && it->second.version == batch[i].version) {
                    valid_keys.push_back(batch[i].group_key);
                } else {
                    stale_keys.push_back(batch[i].group_key);
                }
            }
        }

        if (record_ && !valid_keys.empty()) {
            record_->mark_persisted(valid_keys);
        }
        for (const auto &sk : stale_keys) {
            storage_->remove_group(sk);
        }

        std::unordered_set<std::string> valid_set(valid_keys.begin(), valid_keys.end());
        std::vector<std::string> late_stale_keys;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto &snap : batch) {
                auto it = cache_.find(snap.group_key);
                if (it == cache_.end())
                    continue;
                EntryGroup &group = it->second;
                group.is_persisting = false;
                group.flush_retired_buffers();
                if (valid_set.count(snap.group_key) > 0) {
                    if (group.version == snap.version) {
                        if (!group.is_persisted) {
                            group.is_persisted = true;
                            unpersisted_remove(&group);
                        }
                        auto_persisted_groups_.fetch_add(1, std::memory_order_relaxed);
                        auto_persisted_bytes_.fetch_add(snap.written_bytes,
                                                        std::memory_order_relaxed);
                    } else {
                        late_stale_keys.push_back(snap.group_key);
                    }
                }
            }
        }
        for (const auto &sk : late_stale_keys) {
            storage_->remove_group(sk);
        }

        return true;
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
            size_t step_limit = 1;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                cur_bytes = current_bytes_;
                pace_ms = spill_pace_ms_;
                if (max_groups > 0) {
                    step_limit =
                        std::min<size_t>(spill_batch_groups_, max_groups - groups_processed);
                } else if (use_target_bytes && current_bytes_ > target_bytes && !cache_.empty()) {
                    size_t avg_bytes = std::max<size_t>(1, current_bytes_ / cache_.size());
                    size_t excess = current_bytes_ - target_bytes;
                    size_t est = std::max<size_t>(1, (excess + avg_bytes - 1) / avg_bytes);
                    step_limit = std::min<size_t>(spill_batch_groups_, est);
                } else {
                    step_limit = spill_batch_groups_;
                }
            }
            if (use_target_bytes && cur_bytes <= target_bytes) {
                break;
            }

            if (is_auto_worker && pace_ms > 0 &&
                (active_gets_.load(std::memory_order_relaxed) > 0 ||
                 active_group_readers_total_.load(std::memory_order_relaxed) > 0 ||
                 is_deletion_guarded())) {
                std::this_thread::sleep_for(std::chrono::milliseconds(pace_ms));
            }

            bool made_progress = false;
            if (evict_only) {
                made_progress =
                    spill_step_evict_only(step_limit, is_auto_worker, result, groups_processed);
            } else {
                made_progress = spill_step_persist_then_evict(
                    step_limit, target_bytes, use_target_bytes, max_groups, is_auto_worker, result,
                    groups_processed);
            }

            if (!made_progress) {
                break;
            }
            result.triggered = true;
            result.rounds++;
        }

        return result;
    }

    bool spill_step_persist_then_evict(size_t step_limit, size_t target_bytes,
                                       bool use_target_bytes, size_t max_groups,
                                       bool is_auto_worker, SpillResult &result,
                                       size_t &groups_processed) {
        if (!storage_ || step_limit == 0) {
            return false;
        }

        struct CandidateSnap {
            std::string group_key;
            std::string rep_full_label;
            uint64_t version = 0;
            uint64_t access_seq = 0;
            size_t written_bytes = 0;
            std::vector<StorageBackend::GroupEntryItem> entries;
        };

        std::vector<CandidateSnap> batch;
        batch.reserve(step_limit);
        size_t fast_evicted = 0;

        // Step 1: pick up to step_limit oldest LRU groups under lock.
        // Pre-persisted cold groups are evicted immediately via Fast Path without disk I/O!
        // Unpersisted cold groups are marked is_persisting with zero-copy pointers.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            double now = now_timestamp();
            for (int pass = 0; pass < 2 && batch.empty() && fast_evicted == 0; ++pass) {
                bool allow_incomplete = (pass == 1);
                EntryGroup *group = lru_head_.lru_next;
                while (group != &lru_head_ && (batch.size() + fast_evicted) < step_limit) {
                    if (max_groups > 0 && (groups_processed + batch.size()) >= max_groups) {
                        break;
                    }
                    if (use_target_bytes && current_bytes_ <= target_bytes && batch.empty()) {
                        break;
                    }
                    EntryGroup *next = group->lru_next;
                    bool for_persist = !group->is_persisted;
                    if (is_candidate_eligible_unlocked(*group, allow_incomplete, now,
                                                       for_persist)) {
                        const std::string gk = *group->group_key_ptr;
                        const std::string rep_fl =
                            make_full_label(gk, group->entries.begin()->first);

                        if (group->is_persisted) {
                            if (group->active_readers == 0 && !is_deletion_guarded()) {
                                size_t freed = remove_unlocked(rep_fl);
                                evictions_.fetch_add(1, std::memory_order_relaxed);
                                result.evicted_groups++;
                                result.evicted_bytes += freed;
                                result.evicted_labels.push_back(gk);
                                if (is_auto_worker) {
                                    auto_evicted_groups_.fetch_add(1, std::memory_order_relaxed);
                                    auto_evicted_bytes_.fetch_add(freed, std::memory_order_relaxed);
                                }
                                groups_processed++;
                                fast_evicted++;
                            }
                        } else {
                            group->is_persisting = true;
                            CandidateSnap cand;
                            cand.group_key = gk;
                            cand.rep_full_label = rep_fl;
                            cand.version = group->version;
                            cand.access_seq = group->access_seq;
                            cand.entries.reserve(group->entries.size());
                            for (const auto &[tensor_key, sub] : group->entries) {
                                if (sub.data && sub.size > 0) {
                                    cand.entries.push_back({tensor_key, sub.data, sub.size});
                                    cand.written_bytes += sub.size;
                                }
                            }
                            batch.push_back(std::move(cand));
                        }
                    }
                    group = next;
                }
            }
        }

        if (batch.empty()) {
            return fast_evicted > 0;
        }

        // Step 2: perform aggregated group disk I/O outside lock
        std::vector<StorageBackend::GroupWriteItem> write_groups;
        write_groups.reserve(batch.size());
        for (const auto &cand : batch) {
            write_groups.push_back({cand.group_key, cand.entries});
        }
        std::vector<bool> save_res =
            storage_->save_groups_batch(write_groups, /*allow_parallel=*/!is_auto_worker);

        // Step 3a: check version under lock BEFORE updating Record::mark_persisted
        std::vector<std::string> valid_persisted_keys;
        std::vector<std::string> stale_persisted_keys;
        valid_persisted_keys.reserve(batch.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t i = 0; i < batch.size(); ++i) {
                if (!save_res[i])
                    continue;
                auto it = cache_.find(batch[i].group_key);
                if (it != cache_.end() && it->second.version == batch[i].version) {
                    valid_persisted_keys.push_back(batch[i].group_key);
                } else {
                    stale_persisted_keys.push_back(batch[i].group_key);
                }
            }
        }

        // Step 3b: single batched Record update outside lock for unmodified groups
        if (record_ && !valid_persisted_keys.empty()) {
            record_->mark_persisted(valid_persisted_keys);
        }
        for (const auto &sk : stale_persisted_keys) {
            storage_->remove_group(sk);
        }

        if (is_deletion_guarded()) {
            wait_deletion_guard();
        }

        // Step 3c: verify version & access_seq under lock, then mark persisted and evict if still cold
        std::unordered_set<std::string> valid_set(valid_persisted_keys.begin(),
                                                  valid_persisted_keys.end());
        std::vector<std::string> late_stale_keys;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto &cand : batch) {
                auto it = cache_.find(cand.group_key);
                if (it == cache_.end()) {
                    continue;
                }
                EntryGroup &group = it->second;
                group.is_persisting = false;
                group.flush_retired_buffers();

                if (valid_set.count(cand.group_key) > 0) {
                    if (group.version == cand.version) {
                        if (!group.is_persisted) {
                            group.is_persisted = true;
                            unpersisted_remove(&group);
                        }
                        result.persisted_groups++;
                        result.persisted_bytes += cand.written_bytes;
                        result.persisted_labels.push_back(cand.group_key);
                        if (is_auto_worker) {
                            auto_persisted_groups_.fetch_add(1, std::memory_order_relaxed);
                            auto_persisted_bytes_.fetch_add(cand.written_bytes,
                                                            std::memory_order_relaxed);
                        }
                    } else {
                        late_stale_keys.push_back(cand.group_key);
                    }
                }

                groups_processed++;

                // Evict only if persisted, unmodified, not accessed by GET, and no active readers
                if (group.is_persisted && group.version == cand.version &&
                    group.access_seq == cand.access_seq && group.active_readers == 0 &&
                    !is_deletion_guarded()) {
                    if (!use_target_bytes || current_bytes_ > target_bytes) {
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
            }
        }
        for (const auto &sk : late_stale_keys) {
            storage_->remove_group(sk);
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
                    if (is_candidate_eligible_unlocked(*group, allow_incomplete, now,
                                                       /*for_persist=*/false)) {
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

        if (is_deletion_guarded()) {
            wait_deletion_guard();
        }

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

                if (!is_deletion_guarded() && group.active_readers == 0 &&
                    group.version == item.version && group.access_seq == item.access_seq) {
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

    void put_loaded_group_unlocked(const std::string &group_key,
                                   StorageBackend::LoadedGroup &&loaded_entries,
                                   const std::unordered_set<std::string> &requested_tensor_keys,
                                   double now) {
        for (auto &le : loaded_entries) {
            if (!le.data)
                continue;
            bool is_req = requested_tensor_keys.count(le.tensor_key) > 0;
            std::string full_label = make_full_label(group_key, le.tensor_key);
            put_loaded_one_unlocked(full_label, le.data, le.size, now,
                                    /*prefetched=*/!is_req);
            le.data = nullptr;
            if (is_req) {
                hits_in_storage_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        // Check if any requested tensor_key was missing from the loaded group
        auto it = cache_.find(group_key);
        for (const auto &req_tk : requested_tensor_keys) {
            if (it == cache_.end() || it->second.find(req_tk) == nullptr) {
                misses_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    void put_loaded_one_unlocked(const std::string &full_label, char *data, size_t size,
                                 double now, bool prefetched = false) {
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
                if (!prefetched && sub->prefetched_from_storage) {
                    sub->prefetched_from_storage = false;
                }
            } else {
                SubEntry &new_sub = group.get_or_create(tensor_key);
                new_sub.data = data;
                new_sub.size = size;
                new_sub.prefetched_from_storage = prefetched;
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
            if (expected_group_size_ == 0 || group.entries.size() >= expected_group_size_) {
                group.is_complete = true;
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
            sub.prefetched_from_storage = prefetched;
            group.created_at = now;
            group.last_access = now;
            group.version = 1;
            group.access_seq = ++global_access_seq_;
            group.is_persisted = true;
            group.is_complete =
                (expected_group_size_ == 0 || group.entries.size() >= expected_group_size_);

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
        const bool is_default_kv = (parse_group_key(group_key_sv).first == "kv");

        auto it = cache_.find(group_key);
        if (it != cache_.end()) {

            EntryGroup &group = it->second;
            SubEntry *sub = group.find(tensor_key_sv);
            if (sub && sub->data) {
                if (sub->size == size && std::memcmp(sub->data, data, size) == 0) {
                    // Content-addressed duplicate put with identical payload:
                    // keep existing buffer immutable so active readers/persisters are unaffected.
                    free(data);
                    sub->prefetched_from_storage = false;
                    group.last_access = now;
                    group.access_seq = ++global_access_seq_;
                    if (pre_completed_groups_.find(group_key) != pre_completed_groups_.end()) {
                        if (expected_group_size_ == 0 || !is_default_kv ||
                            group.entries.size() >= expected_group_size_) {
                            pre_completed_groups_.erase(group_key);
                            group.is_complete = true;
                        }
                    } else if (expected_group_size_ > 0 && is_default_kv &&
                               group.entries.size() >= expected_group_size_) {
                        group.is_complete = true;
                    }
                    lru_remove(&group);
                    lru_push_back(&group);
                    puts_.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                current_bytes_ -= sub->size;
                if (group.active_readers > 0 || group.is_persisting || is_deletion_guarded()) {
                    group.retired_buffers.push_back(sub->data);
                } else {
                    free(sub->data);
                }
                sub->data = data;
                sub->size = size;
                sub->prefetched_from_storage = false;
            } else {
                SubEntry &new_sub = group.get_or_create(tensor_key);
                new_sub.data = data;
                new_sub.size = size;
                new_sub.prefetched_from_storage = false;
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
            if (pre_completed_groups_.find(group_key) != pre_completed_groups_.end()) {
                if (expected_group_size_ == 0 || !is_default_kv ||
                    group.entries.size() >= expected_group_size_) {
                    pre_completed_groups_.erase(group_key);
                    group.is_complete = true;
                }
            } else if (expected_group_size_ > 0 && is_default_kv &&
                       group.entries.size() >= expected_group_size_) {
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
            sub.prefetched_from_storage = false;
            group.created_at = now;
            group.last_access = now;
            group.version = 1;
            group.access_seq = ++global_access_seq_;
            group.is_persisted = false;
            if (pre_completed_groups_.find(group_key) != pre_completed_groups_.end()) {
                if (expected_group_size_ == 0 || !is_default_kv ||
                    group.entries.size() >= expected_group_size_) {
                    pre_completed_groups_.erase(group_key);
                    group.is_complete = true;
                }
            } else if (expected_group_size_ > 0 && is_default_kv &&
                       group.entries.size() >= expected_group_size_) {
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

    std::pair<const char *, size_t> get_one_unlocked(const std::string &full_label, double now,
                                                     EntryGroup **out_group = nullptr) {
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
                if (sub->prefetched_from_storage) {
                    sub->prefetched_from_storage = false;
                    hits_in_storage_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    hits_.fetch_add(1, std::memory_order_relaxed);
                }
                if (out_group) {
                    *out_group = &group;
                }
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
            if (curr->is_persisting || curr->is_unpublishing || curr->active_readers > 0) {
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
    size_t expected_group_size_ = 0;
    uint64_t global_access_seq_ = 0;

    EntryGroup lru_head_;
    EntryGroup unpersisted_head_;

    size_t capacity_bytes_;
    size_t current_bytes_;
    size_t total_entries_;

    std::atomic<bool> evict_enabled_;

    mutable std::mutex mutex_;

    std::atomic<int> deletion_guard_count_{0};
    std::atomic<int> active_group_readers_total_{0};
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

void Mem::set_expected_group_size(size_t expected_size) {
    impl_->set_expected_group_size(expected_size);
}

size_t Mem::expected_group_size() const { return impl_->expected_group_size(); }

void Mem::release_group_guards(const std::vector<std::string> &keys) {
    impl_->release_group_guards(keys);
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
