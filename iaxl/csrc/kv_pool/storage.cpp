// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "env.h"
#include "kv_pool.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/uio.h>
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

bool StorageBackend::save_group(const std::string &group_key,
                                const std::vector<GroupEntryItem> &entries) {
    if (entries.empty()) {
        return false;
    }
    for (const auto &entry : entries) {
        if (!save(make_full_label(group_key, entry.tensor_key), entry.data, entry.size)) {
            return false;
        }
    }
    return true;
}

StorageBackend::LoadedGroup StorageBackend::load_group(const std::string &group_key) {
    (void)group_key;
    return {};
}

bool StorageBackend::remove_group(const std::string &group_key) {
    (void)group_key;
    return true;
}

std::vector<bool> StorageBackend::save_groups_batch(const std::vector<GroupWriteItem> &groups,
                                                    bool allow_parallel) {
    std::vector<bool> results(groups.size(), false);
    if (groups.empty()) {
        return results;
    }
    if (allow_parallel && groups.size() > 1) {
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < groups.size(); ++i) {
            results[i] = save_group(groups[i].group_key, groups[i].entries);
        }
    } else {
        for (size_t i = 0; i < groups.size(); ++i) {
            results[i] = save_group(groups[i].group_key, groups[i].entries);
        }
    }
    return results;
}

std::vector<StorageBackend::LoadedGroup>
StorageBackend::load_groups_batch(const std::vector<std::string> &group_keys, bool allow_parallel) {
    std::vector<LoadedGroup> results(group_keys.size());
    if (group_keys.empty()) {
        return results;
    }
    if (allow_parallel && group_keys.size() > 1) {
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < group_keys.size(); ++i) {
            results[i] = load_group(group_keys[i]);
        }
    } else {
        for (size_t i = 0; i < group_keys.size(); ++i) {
            results[i] = load_group(group_keys[i]);
        }
    }
    return results;
}

namespace {

constexpr char kGroupFileMagic[4] = {'K', 'V', 'G', '1'};
constexpr int kMaxIovChunk = 512;

bool writev_all(int fd, std::vector<struct iovec> &iovs, size_t expected_total) {
    size_t written_total = 0;
    size_t idx = 0;
    while (idx < iovs.size()) {
        int cnt = static_cast<int>(std::min<size_t>(iovs.size() - idx, kMaxIovChunk));
        ssize_t n = ::writev(fd, &iovs[idx], cnt);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        written_total += static_cast<size_t>(n);
        size_t rem = static_cast<size_t>(n);
        while (rem > 0 && idx < iovs.size()) {
            if (rem >= iovs[idx].iov_len) {
                rem -= iovs[idx].iov_len;
                iovs[idx].iov_len = 0;
                idx++;
            } else {
                iovs[idx].iov_base = static_cast<char *>(iovs[idx].iov_base) + rem;
                iovs[idx].iov_len -= rem;
                rem = 0;
            }
        }
    }
    return written_total == expected_total;
}

bool readv_all(int fd, std::vector<struct iovec> &iovs, size_t expected_total) {
    size_t read_total = 0;
    size_t idx = 0;
    while (idx < iovs.size()) {
        if (iovs[idx].iov_len == 0) {
            idx++;
            continue;
        }
        int cnt = static_cast<int>(std::min<size_t>(iovs.size() - idx, kMaxIovChunk));
        ssize_t n = ::readv(fd, &iovs[idx], cnt);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            break;
        }
        read_total += static_cast<size_t>(n);
        size_t rem = static_cast<size_t>(n);
        while (rem > 0 && idx < iovs.size()) {
            if (rem >= iovs[idx].iov_len) {
                rem -= iovs[idx].iov_len;
                iovs[idx].iov_len = 0;
                idx++;
            } else {
                iovs[idx].iov_base = static_cast<char *>(iovs[idx].iov_base) + rem;
                iovs[idx].iov_len -= rem;
                rem = 0;
            }
        }
    }
    return read_total == expected_total;
}

bool pread_exact(int fd, char *buf, size_t len, off_t offset) {
    size_t done = 0;
    while (done < len) {
        ssize_t n = ::pread(fd, buf + done, len - done, offset + static_cast<off_t>(done));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

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

    bool save_group(const std::string &group_key,
                    const std::vector<GroupEntryItem> &entries) override {
        if (entries.empty()) {
            return false;
        }

        std::string path_str = get_group_path_from_key(group_key);
        std::filesystem::path file_path =
            std::filesystem::path(persist_dir_) / "chunks" / (path_str + ".grp.bin");
        std::filesystem::path dir_path = file_path.parent_path();

        std::error_code ec;
        std::filesystem::create_directories(dir_path, ec);
        if (ec) {
            std::cerr << "\033[1;31m*** FATAL: error in creating directory " << dir_path << ": "
                      << ec.message() << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        uint32_t header_bytes = 12;
        for (const auto &entry : entries) {
            header_bytes += 4 + 8 + static_cast<uint32_t>(entry.tensor_key.size());
        }

        std::vector<char> header(header_bytes);
        std::memcpy(header.data(), kGroupFileMagic, 4);
        uint32_t num_entries = static_cast<uint32_t>(entries.size());
        std::memcpy(header.data() + 4, &num_entries, 4);
        std::memcpy(header.data() + 8, &header_bytes, 4);

        size_t pos = 12;
        size_t payload_bytes = 0;
        for (const auto &entry : entries) {
            uint32_t klen = static_cast<uint32_t>(entry.tensor_key.size());
            uint64_t dsize = static_cast<uint64_t>(entry.size);
            std::memcpy(header.data() + pos, &klen, 4);
            pos += 4;
            std::memcpy(header.data() + pos, &dsize, 8);
            pos += 8;
            if (klen > 0) {
                std::memcpy(header.data() + pos, entry.tensor_key.data(), klen);
                pos += klen;
            }
            payload_bytes += entry.size;
        }

        static std::atomic<uint64_t> tmp_seq{0};
        const uint64_t seq = tmp_seq.fetch_add(1, std::memory_order_relaxed);
        std::filesystem::path tmp_path =
            file_path.string() + ".tmp." + std::to_string(getpid()) + "_" + std::to_string(seq);

        int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0) {
            std::cerr << "\033[1;31m*** FATAL: Error opening group file for writing: " << tmp_path
                      << " (" << std::strerror(errno) << ")\033[0m\n"
                      << std::endl;
            std::abort();
        }

        std::vector<struct iovec> iovs;
        iovs.reserve(entries.size() + 1);
        iovs.push_back({header.data(), header.size()});
        for (const auto &entry : entries) {
            if (entry.data && entry.size > 0) {
                iovs.push_back({const_cast<char *>(entry.data), entry.size});
            }
        }

        const size_t expected_total = header.size() + payload_bytes;
        if (!writev_all(fd, iovs, expected_total)) {
            ::close(fd);
            std::filesystem::remove(tmp_path, ec);
            std::cerr << "\033[1;31m*** FATAL: Error writing group file: " << tmp_path
                      << "\033[0m\n"
                      << std::endl;
            std::abort();
        }
        ::close(fd);

        std::filesystem::rename(tmp_path, file_path, ec);
        if (ec) {
            std::filesystem::remove(tmp_path, ec);
            std::cerr << "\033[1;31m*** FATAL: Error renaming temp group file " << tmp_path
                      << " to " << file_path << ": " << ec.message() << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        if (envs.IAXL_DEBUG_LOG) {
            std::cout << "[storage] Saved group " << group_key << " (" << entries.size()
                      << " entries, " << expected_total << " bytes) to " << file_path << "\n";
        }
        return true;
    }

    LoadedGroup load_group(const std::string &group_key) override {
        std::string path_str = get_group_path_from_key(group_key);
        std::filesystem::path file_path =
            std::filesystem::path(persist_dir_) / "chunks" / (path_str + ".grp.bin");

        int fd = ::open(file_path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            if (errno == ENOENT) {
                return {};
            }
            std::cerr << "\033[1;31m*** FATAL: Error opening group file for reading: " << file_path
                      << " (" << std::strerror(errno) << ")\033[0m\n"
                      << std::endl;
            std::abort();
        }

        char prefix[12];
        if (!pread_exact(fd, prefix, sizeof(prefix), 0) ||
            std::memcmp(prefix, kGroupFileMagic, 4) != 0) {
            ::close(fd);
            std::cerr << "\033[1;31m*** FATAL: Invalid group file header: " << file_path
                      << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        uint32_t num_entries = 0;
        uint32_t header_bytes = 0;
        std::memcpy(&num_entries, prefix + 4, 4);
        std::memcpy(&header_bytes, prefix + 8, 4);
        if (num_entries == 0 || header_bytes < 12) {
            ::close(fd);
            std::cerr << "\033[1;31m*** FATAL: Corrupted group file metadata: " << file_path
                      << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        std::vector<char> index_buf(header_bytes - 12);
        if (!index_buf.empty() && !pread_exact(fd, index_buf.data(), index_buf.size(), 12)) {
            ::close(fd);
            std::cerr << "\033[1;31m*** FATAL: Failed reading group index table: " << file_path
                      << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        LoadedGroup result;
        result.reserve(num_entries);
        std::vector<struct iovec> iovs;
        iovs.reserve(num_entries);

        size_t pos = 0;
        size_t expected_payload = 0;
        for (uint32_t i = 0; i < num_entries; ++i) {
            if (pos + 12 > index_buf.size()) {
                for (auto &e : result) {
                    std::free(e.data);
                }
                ::close(fd);
                std::cerr << "\033[1;31m*** FATAL: Truncated group index table: " << file_path
                          << "\033[0m\n"
                          << std::endl;
                std::abort();
            }
            uint32_t klen = 0;
            uint64_t dsize = 0;
            std::memcpy(&klen, index_buf.data() + pos, 4);
            pos += 4;
            std::memcpy(&dsize, index_buf.data() + pos, 8);
            pos += 8;
            if (pos + klen > index_buf.size()) {
                for (auto &e : result) {
                    std::free(e.data);
                }
                ::close(fd);
                std::cerr << "\033[1;31m*** FATAL: Corrupt key in group index table: " << file_path
                          << "\033[0m\n"
                          << std::endl;
                std::abort();
            }
            std::string tensor_key(index_buf.data() + pos, klen);
            pos += klen;

            char *buf = nullptr;
            if (dsize > 0) {
                buf = static_cast<char *>(std::malloc(static_cast<size_t>(dsize)));
                if (!buf) {
                    for (auto &e : result) {
                        std::free(e.data);
                    }
                    ::close(fd);
                    std::cerr << "\033[1;31m*** FATAL: Memory allocation failed for size: "
                              << dsize << "\033[0m\n"
                              << std::endl;
                    std::abort();
                }
                iovs.push_back({buf, static_cast<size_t>(dsize)});
                expected_payload += static_cast<size_t>(dsize);
            }
            result.push_back({std::move(tensor_key), buf, static_cast<size_t>(dsize)});
        }

        if (::lseek(fd, static_cast<off_t>(header_bytes), SEEK_SET) < 0 ||
            !readv_all(fd, iovs, expected_payload)) {
            for (auto &e : result) {
                std::free(e.data);
            }
            ::close(fd);
            std::cerr << "\033[1;31m*** FATAL: Failed reading group payload: " << file_path
                      << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        ::close(fd);
        return result;
    }

    bool remove_group(const std::string &group_key) override {
        std::string path_str = get_group_path_from_key(group_key);
        std::filesystem::path file_path =
            std::filesystem::path(persist_dir_) / "chunks" / (path_str + ".grp.bin");
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

bool Storage::save_group(const std::string &group_key,
                         const std::vector<StorageBackend::GroupEntryItem> &entries) {
    return backend_->save_group(group_key, entries);
}

StorageBackend::LoadedGroup Storage::load_group(const std::string &group_key) {
    return backend_->load_group(group_key);
}

bool Storage::remove_group(const std::string &group_key) {
    return backend_->remove_group(group_key);
}

std::vector<bool>
Storage::save_groups_batch(const std::vector<StorageBackend::GroupWriteItem> &groups,
                           bool allow_parallel) {
    return backend_->save_groups_batch(groups, allow_parallel);
}

std::vector<StorageBackend::LoadedGroup>
Storage::load_groups_batch(const std::vector<std::string> &group_keys, bool allow_parallel) {
    return backend_->load_groups_batch(group_keys, allow_parallel);
}

} // namespace kv_pool
