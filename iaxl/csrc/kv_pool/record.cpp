// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "kv_pool.h"
#include "task_queue.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <iostream>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sqlite3.h>

#define SQLITE_CHECK(call)                                                                         \
    do {                                                                                           \
        int rc_ = (call);                                                                          \
        if (rc_ != SQLITE_OK) {                                                                    \
            fprintf(stderr, "SQLite error %s:%d: %s returned %d\n", __FILE__, __LINE__, #call,     \
                    rc_);                                                                          \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

#define SQLITE_CHECK_DONE(call)                                                                    \
    do {                                                                                           \
        int rc_ = (call);                                                                          \
        if (rc_ != SQLITE_DONE) {                                                                  \
            fprintf(stderr, "SQLite error %s:%d: %s returned %d\n", __FILE__, __LINE__, #call,     \
                    rc_);                                                                          \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

namespace kv_pool {

namespace {

double now_seconds() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

} // namespace

class Record::Impl {
  public:
    Impl(const std::string &sqlite_path, bool cleanup_unpersisted)
        : sqlite_path_(sqlite_path), db_(nullptr), read_db_(nullptr), stmt_(nullptr),
          queue_("TZ-REC") {

        int rc = sqlite3_open_v2(sqlite_path_.c_str(), &db_,
                                 SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
                                 nullptr);
        if (rc != SQLITE_OK) {
            std::string err = "[Record] Failed to open SQLite: " + std::string(sqlite3_errmsg(db_));
            throw std::runtime_error(err);
        }

        SQLITE_CHECK(sqlite3_exec(db_, "PRAGMA journal_mode=WAL", nullptr, nullptr, nullptr));
        SQLITE_CHECK(sqlite3_exec(db_, "PRAGMA synchronous=OFF", nullptr, nullptr, nullptr));
        SQLITE_CHECK(sqlite3_exec(db_, "PRAGMA cache_size=-1048576", nullptr, nullptr, nullptr));
        SQLITE_CHECK(sqlite3_exec(db_, "PRAGMA temp_store=MEMORY", nullptr, nullptr, nullptr));
        SQLITE_CHECK(sqlite3_exec(db_, "PRAGMA mmap_size=1073741824", nullptr, nullptr, nullptr));
        sqlite3_busy_timeout(db_, 60000);

        const char *create_sql = "CREATE TABLE IF NOT EXISTS chunks ("
                                 "  chunk_label TEXT PRIMARY KEY,"
                                 "  created_at REAL,"
                                 "  persisted INTEGER DEFAULT 0"
                                 ")";
        rc = sqlite3_exec(db_, create_sql, nullptr, nullptr, nullptr);
        if (rc != SQLITE_OK) {
            std::string err =
                "[Record] Failed to create table: " + std::string(sqlite3_errmsg(db_));
            sqlite3_close(db_);
            throw std::runtime_error(err);
        }

        if (cleanup_unpersisted) {
            int changes_before = sqlite3_total_changes(db_);
            SQLITE_CHECK(sqlite3_exec(db_, "DELETE FROM chunks WHERE persisted = 0", nullptr,
                                      nullptr, nullptr));
            int deleted = sqlite3_total_changes(db_) - changes_before;
            if (deleted > 0) {
                std::cerr << "[Record] Cleaned up " << deleted
                          << " unpersisted records from previous run" << std::endl;
            }
        }

        const char *insert_sql =
            "INSERT INTO chunks (chunk_label, created_at, persisted) VALUES (?, ?, 0) "
            "ON CONFLICT(chunk_label) DO UPDATE SET created_at = excluded.created_at";
        SQLITE_CHECK(sqlite3_prepare_v2(db_, insert_sql, -1, &stmt_, nullptr));

        rc = sqlite3_open_v2(sqlite_path_.c_str(), &read_db_,
                             SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
                             nullptr);
        if (rc != SQLITE_OK) {
            std::string err =
                "[Record] Failed to open read SQLite: " + std::string(sqlite3_errmsg(read_db_));
            sqlite3_finalize(stmt_);
            sqlite3_close(db_);
            throw std::runtime_error(err);
        }
        SQLITE_CHECK(sqlite3_exec(read_db_, "PRAGMA journal_mode=WAL", nullptr, nullptr, nullptr));
        SQLITE_CHECK(sqlite3_exec(read_db_, "PRAGMA cache_size=-262144", nullptr, nullptr, nullptr));
        SQLITE_CHECK(sqlite3_exec(read_db_, "PRAGMA temp_store=MEMORY", nullptr, nullptr, nullptr));
        SQLITE_CHECK(
            sqlite3_exec(read_db_, "PRAGMA mmap_size=1073741824", nullptr, nullptr, nullptr));
        sqlite3_busy_timeout(read_db_, 60000);

        queue_.init();
    }

    ~Impl() { shutdown(); }

    void submit(const std::string &label, const std::vector<std::string> &chunk_labels) {
        auto group_keys = make_group_keys(label, chunk_labels);
        std::lock_guard<std::mutex> lock(operation_mutex_);

        auto future = queue_.submit([this, group_keys = std::move(group_keys)]() {
            double current_time = now_seconds();

            SQLITE_CHECK(sqlite3_exec(db_, "BEGIN TRANSACTION", nullptr, nullptr, nullptr));

            constexpr size_t batch_size = 900;
            for (size_t i = 0; i < group_keys.size(); i += batch_size) {
                size_t end = std::min(i + batch_size, group_keys.size());
                for (size_t j = i; j < end; j++) {
                    sqlite3_reset(stmt_);
                    sqlite3_bind_text(stmt_, 1, group_keys[j].c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_double(stmt_, 2, current_time);

                    SQLITE_CHECK_DONE(sqlite3_step(stmt_));
                }
            }

            SQLITE_CHECK(sqlite3_exec(db_, "COMMIT", nullptr, nullptr, nullptr));
        });

        track_future_locked(future.share());
    }

    void mark_persisted(const std::vector<std::string> &chunk_labels) {
        if (chunk_labels.empty())
            return;
        std::shared_future<void> shared_fut;
        {
            std::lock_guard<std::mutex> lock(operation_mutex_);

            auto future = queue_.submit([this, chunk_labels = chunk_labels]() {
                double current_time = now_seconds();
                SQLITE_CHECK(
                    sqlite3_exec(db_, "PRAGMA synchronous=NORMAL", nullptr, nullptr, nullptr));

                sqlite3_stmt *update_stmt = nullptr;
                const char *update_sql =
                    "INSERT INTO chunks (chunk_label, created_at, persisted) VALUES (?, ?, 1) "
                    "ON CONFLICT(chunk_label) DO UPDATE SET persisted = 1";

                SQLITE_CHECK(sqlite3_prepare_v2(db_, update_sql, -1, &update_stmt, nullptr));
                SQLITE_CHECK(sqlite3_exec(db_, "BEGIN TRANSACTION", nullptr, nullptr, nullptr));

                for (const auto &gk : chunk_labels) {
                    sqlite3_reset(update_stmt);
                    sqlite3_bind_text(update_stmt, 1, gk.c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_double(update_stmt, 2, current_time);

                    SQLITE_CHECK_DONE(sqlite3_step(update_stmt));
                }

                SQLITE_CHECK(sqlite3_exec(db_, "COMMIT", nullptr, nullptr, nullptr));
                sqlite3_finalize(update_stmt);

                SQLITE_CHECK(
                    sqlite3_exec(db_, "PRAGMA synchronous=OFF", nullptr, nullptr, nullptr));
            });

            shared_fut = future.share();
            track_future_locked(shared_fut);
        }
        shared_fut.get();
    }

    void remove(const std::vector<std::string> &chunk_labels) {
        if (chunk_labels.empty())
            return;
        std::shared_future<void> shared_fut;
        {
            std::lock_guard<std::mutex> lock(operation_mutex_);

            auto future = queue_.submit([this, chunk_labels]() {
                sqlite3_stmt *delete_stmt = nullptr;
                const char *delete_sql = "DELETE FROM chunks WHERE chunk_label = ?";

                SQLITE_CHECK(sqlite3_prepare_v2(db_, delete_sql, -1, &delete_stmt, nullptr));
                SQLITE_CHECK(sqlite3_exec(db_, "BEGIN TRANSACTION", nullptr, nullptr, nullptr));

                for (const auto &label : chunk_labels) {
                    sqlite3_reset(delete_stmt);
                    sqlite3_bind_text(delete_stmt, 1, label.c_str(), -1, SQLITE_TRANSIENT);

                    SQLITE_CHECK_DONE(sqlite3_step(delete_stmt));
                }

                SQLITE_CHECK(sqlite3_exec(db_, "COMMIT", nullptr, nullptr, nullptr));
                sqlite3_finalize(delete_stmt);
            });

            shared_fut = future.share();
            track_future_locked(shared_fut);
        }
        shared_fut.get();
    }

    std::vector<bool> is_persisted(const std::vector<std::string> &chunk_labels) {
        if (chunk_labels.empty())
            return {};
        std::lock_guard<std::mutex> rlock(read_mutex_);
        assert(read_db_ && "[Record] read_db_ must not be null");

        std::vector<bool> results(chunk_labels.size(), false);
        std::unordered_map<std::string, bool> persisted_status;
        persisted_status.reserve(chunk_labels.size());

        constexpr size_t batch_size = 900;
        for (size_t i = 0; i < chunk_labels.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, chunk_labels.size());
            size_t count = end - i;

            std::string query = "SELECT chunk_label, persisted FROM chunks WHERE chunk_label IN (";
            for (size_t j = 0; j < count; j++) {
                if (j > 0)
                    query += ',';
                query += '?';
            }
            query += ')';

            sqlite3_stmt *select_stmt = nullptr;
            SQLITE_CHECK(sqlite3_prepare_v2(read_db_, query.c_str(), -1, &select_stmt, nullptr));

            for (size_t j = 0; j < count; j++) {
                sqlite3_bind_text(select_stmt, j + 1, chunk_labels[i + j].c_str(), -1,
                                  SQLITE_TRANSIENT);
            }

            while (sqlite3_step(select_stmt) == SQLITE_ROW) {
                const char *label = (const char *)sqlite3_column_text(select_stmt, 0);
                int persisted = sqlite3_column_int(select_stmt, 1);
                if (label) {
                    persisted_status[label] = (persisted == 1);
                }
            }
            sqlite3_finalize(select_stmt);
        }

        for (size_t i = 0; i < chunk_labels.size(); ++i) {
            auto it = persisted_status.find(chunk_labels[i]);
            if (it != persisted_status.end()) {
                results[i] = it->second;
            }
        }

        return results;
    }

    void sync() {
        std::list<std::shared_future<void>> futures_copy;
        {
            std::lock_guard<std::mutex> lock(operation_mutex_);
            futures_copy = std::move(futures_);
            futures_.clear();
        }
        wait_futures(futures_copy);
    }

    std::vector<bool> has(const std::string &label, const std::vector<std::string> &chunk_labels) {
        if (chunk_labels.empty())
            return {};
        sync();

        std::lock_guard<std::mutex> rlock(read_mutex_);
        assert(read_db_ && "[Record] read_db_ must not be null");

        auto group_keys = make_group_keys(label, chunk_labels);

        std::vector<bool> results(chunk_labels.size(), false);
        constexpr size_t batch_size = 900;

        for (size_t i = 0; i < group_keys.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, group_keys.size());
            size_t count = end - i;

            std::string query = "SELECT chunk_label FROM chunks WHERE chunk_label IN (";
            for (size_t j = 0; j < count; j++) {
                if (j > 0)
                    query += ',';
                query += '?';
            }
            query += ')';

            sqlite3_stmt *select_stmt = nullptr;
            int rc = sqlite3_prepare_v2(read_db_, query.c_str(), -1, &select_stmt, nullptr);
            assert(rc == SQLITE_OK && "[Record] has() prepare failed");
            (void)rc;

            for (size_t j = 0; j < count; j++) {
                sqlite3_bind_text(select_stmt, j + 1, group_keys[i + j].c_str(), -1,
                                  SQLITE_TRANSIENT);
            }

            std::unordered_set<std::string> exists_set;
            exists_set.reserve(count);
            while (sqlite3_step(select_stmt) == SQLITE_ROW) {
                const char *text = (const char *)sqlite3_column_text(select_stmt, 0);
                if (text)
                    exists_set.insert(text);
            }
            sqlite3_finalize(select_stmt);

            for (size_t j = 0; j < count; j++) {
                results[i + j] = exists_set.count(group_keys[i + j]) > 0;
            }
        }

        return results;
    }

    void shutdown() {
        sync();
        std::lock_guard<std::mutex> lock(operation_mutex_);
        std::lock_guard<std::mutex> rlock(read_mutex_);
        if (!db_ && !read_db_)
            return;

        queue_.shutdown();

        if (stmt_) {
            sqlite3_finalize(stmt_);
            stmt_ = nullptr;
        }

        if (read_db_) {
            sqlite3_close(read_db_);
            read_db_ = nullptr;
        }

        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
    }

    size_t pending_count() const {
        std::lock_guard<std::mutex> lock(operation_mutex_);
        size_t active = 0;
        for (const auto &fut : futures_) {
            if (fut.valid() && fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                ++active;
            }
        }
        return active;
    }

  private:
    static void wait_futures(std::list<std::shared_future<void>> &futures_list) {
        std::exception_ptr first_error;
        for (auto &future : futures_list) {
            if (!future.valid())
                continue;
            try {
                future.get();
            } catch (...) {
                if (!first_error)
                    first_error = std::current_exception();
            }
        }
        if (first_error)
            std::rethrow_exception(first_error);
    }

    void track_future_locked(std::shared_future<void> future) {
        futures_.push_back(std::move(future));

        std::exception_ptr first_error;
        for (auto it = futures_.begin(); it != futures_.end();) {
            if (it->wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                ++it;
                continue;
            }
            try {
                it->get();
            } catch (...) {
                if (!first_error)
                    first_error = std::current_exception();
            }
            it = futures_.erase(it);
        }
        if (first_error)
            std::rethrow_exception(first_error);
    }

    std::string sqlite_path_;
    sqlite3 *db_;
    sqlite3 *read_db_;
    sqlite3_stmt *stmt_;
    TaskQueue queue_;

    mutable std::mutex operation_mutex_;
    mutable std::mutex read_mutex_;
    std::list<std::shared_future<void>> futures_;
};

Record::Record(const std::string &sqlite_path, bool cleanup_unpersisted)
    : impl_(std::make_unique<Impl>(sqlite_path, cleanup_unpersisted)) {}

Record::~Record() = default;

void Record::submit(const std::string &label, const std::vector<std::string> &chunk_labels) {
    impl_->submit(label, chunk_labels);
}

void Record::mark_persisted(const std::vector<std::string> &chunk_labels) {
    impl_->mark_persisted(chunk_labels);
}

void Record::remove(const std::vector<std::string> &chunk_labels) { impl_->remove(chunk_labels); }

std::vector<bool> Record::is_persisted(const std::vector<std::string> &chunk_labels) {
    return impl_->is_persisted(chunk_labels);
}

void Record::sync() { impl_->sync(); }

std::vector<bool> Record::has(const std::string &label,
                              const std::vector<std::string> &chunk_labels) {
    return impl_->has(label, chunk_labels);
}

void Record::shutdown() { impl_->shutdown(); }

size_t Record::pending_count() const { return impl_->pending_count(); }

} // namespace kv_pool
