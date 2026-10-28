// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "env.h"
#include "kv_pool.h"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace kv_pool {

std::vector<bool> StorageBackend::save_batch(const std::vector<WriteItem> &items) {
    std::vector<bool> results(items.size(), false);
    for (size_t i = 0; i < items.size(); ++i) {
        results[i] = save(items[i].full_label, items[i].data, items[i].size);
    }
    return results;
}

std::vector<std::pair<char *, size_t>>
StorageBackend::load_batch(const std::vector<std::string> &full_labels, bool allow_parallel) {
    std::vector<std::pair<char *, size_t>> results(full_labels.size(), {nullptr, 0});
    if (full_labels.empty()) {
        return results;
    }
    if (allow_parallel && full_labels.size() > 1) {
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < full_labels.size(); ++i) {
            results[i] = load(full_labels[i]);
        }
    } else {
        for (size_t i = 0; i < full_labels.size(); ++i) {
            results[i] = load(full_labels[i]);
        }
    }
    return results;
}

namespace {

class FileStorageBackend final : public StorageBackend {
  public:
    explicit FileStorageBackend(const std::string &persist_dir) : persist_dir_(persist_dir) {
        if (persist_dir_.empty()) {
            std::cerr << "\033[1;31m*** FATAL: persist_dir is not set, cannot save data\033[0m\n";
            std::abort();
        }
        if (envs.IAXL_DEBUG_LOG && !persist_dir_.empty()) {
            std::cout << "[storage] Initialized with persist_dir: " << persist_dir_ << "\n";
        }
    }

    bool save(const std::string &full_label, const char *data, size_t size) override {
        std::string path_str = get_path_from_label(full_label);

        std::filesystem::path file_path =
            std::filesystem::path(persist_dir_) / "chunks" / (path_str + ".bin");
        std::filesystem::path dir_path = file_path.parent_path();

        std::error_code ec;
        std::filesystem::create_directories(dir_path, ec);
        if (ec) {
            std::cerr << "\033[1;31m*** FATAL: error in creating directory " << dir_path << ": "
                      << ec.message() << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        static std::atomic<uint64_t> tmp_seq{0};
        const uint64_t seq = tmp_seq.fetch_add(1, std::memory_order_relaxed);
        std::filesystem::path tmp_path =
            file_path.string() + ".tmp." + std::to_string(getpid()) + "_" + std::to_string(seq);

        std::ofstream file(tmp_path, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            std::cerr << "\033[1;31m*** FATAL: Error opening file for writing: " << tmp_path
                      << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        file.write(data, size);
        if (file.tellp() != static_cast<std::streampos>(size)) {
            file.close();
            std::filesystem::remove(tmp_path, ec);
            std::cerr << "\033[1;31m*** FATAL: Error writing file: " << tmp_path << ", wrote "
                      << file.tellp() << " bytes, expected " << size << "\033[0m\n"
                      << std::endl;
            std::abort();
        }
        file.close();

        std::filesystem::rename(tmp_path, file_path, ec);
        if (ec) {
            std::filesystem::remove(tmp_path, ec);
            std::cerr << "\033[1;31m*** FATAL: Error renaming temp file " << tmp_path << " to "
                      << file_path << ": " << ec.message() << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        if (envs.IAXL_DEBUG_LOG) {
            std::cout << "[storage] Saved " << size << " bytes to " << file_path << "\n";
        }

        return true;
    }

    std::pair<char *, size_t> load(const std::string &full_label) override {
        std::string path_str = get_path_from_label(full_label);

        std::filesystem::path file_path =
            std::filesystem::path(persist_dir_) / "chunks" / (path_str + ".bin");

        if (envs.IAXL_DEBUG_LOG) {
            std::cout << "[storage] Attempting to load from file: " << file_path << "\n";
        }

        if (!std::filesystem::exists(file_path)) {
            std::cerr << "\033[1;31m*** FATAL: File not found: " << file_path << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        std::ifstream file(file_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "\033[1;31m*** FATAL: Error opening file for reading: " << file_path
                      << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        char *buffer = (char *)malloc(size);
        if (!buffer) {
            std::cerr << "\033[1;31m*** FATAL: Memory allocation failed for size: " << size
                      << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        if (file.read(buffer, size)) {
            if (file.gcount() != size) {
                free(buffer);
                std::cerr << "\033[1;31m*** FATAL: Error reading file: " << file_path << ", read "
                          << file.gcount() << " bytes, expected " << size << "\033[0m\n"
                          << std::endl;
                std::abort();
            }
            return {buffer, static_cast<size_t>(size)};
        } else {
            free(buffer);
            std::cerr << "\033[1;31m*** FATAL: Error reading file: " << file_path << "\033[0m\n"
                      << std::endl;
            std::abort();
        }
    }

    bool remove(const std::string &full_label) override {
        std::string path_str = get_path_from_label(full_label);
        std::filesystem::path file_path =
            std::filesystem::path(persist_dir_) / "chunks" / (path_str + ".bin");
        std::error_code ec;
        return std::filesystem::remove(file_path, ec);
    }

  private:
    std::string persist_dir_;
};

} // namespace

Storage::Storage(const std::string &persist_dir)
    : backend_(std::make_unique<FileStorageBackend>(persist_dir)) {}

Storage::Storage(std::unique_ptr<StorageBackend> backend) : backend_(std::move(backend)) {
    if (!backend_) {
        std::cerr << "\033[1;31m*** FATAL: Storage backend must not be null\033[0m\n";
        std::abort();
    }
}

Storage::~Storage() = default;

bool Storage::save(const std::string &full_label, const char *data, size_t size) {
    return backend_->save(full_label, data, size);
}

std::pair<char *, size_t> Storage::load(const std::string &full_label) {
    return backend_->load(full_label);
}

bool Storage::remove(const std::string &full_label) { return backend_->remove(full_label); }

std::vector<bool> Storage::save_batch(const std::vector<StorageBackend::WriteItem> &items) {
    return backend_->save_batch(items);
}

std::vector<std::pair<char *, size_t>>
Storage::load_batch(const std::vector<std::string> &full_labels, bool allow_parallel) {
    return backend_->load_batch(full_labels, allow_parallel);
}

} // namespace kv_pool
