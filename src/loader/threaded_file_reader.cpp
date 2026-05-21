#include "threaded_file_reader.h"

#include "loader_stats.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <Windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace sd {
namespace loader {

namespace {

struct ReadChunk {
    uint64_t file_offset = 0;
    size_t dst_offset = 0;
    size_t bytes = 0;
};

bool read_chunk(const std::string& path, const ReadChunk& chunk, char* dst) {
#if defined(_WIN32)
    HANDLE file = CreateFileA(path.c_str(),
                              GENERIC_READ,
                              FILE_SHARE_READ,
                              nullptr,
                              OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
                              nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }
    LARGE_INTEGER pos{};
    pos.QuadPart = static_cast<LONGLONG>(chunk.file_offset);
    if (!SetFilePointerEx(file, pos, nullptr, FILE_BEGIN)) {
        CloseHandle(file);
        return false;
    }
    size_t total = 0;
    while (total < chunk.bytes) {
        const DWORD request = static_cast<DWORD>(std::min<size_t>(chunk.bytes - total, 64ull * 1024ull * 1024ull));
        DWORD read = 0;
        if (!ReadFile(file, dst + chunk.dst_offset + total, request, &read, nullptr) || read == 0) {
            CloseHandle(file);
            return false;
        }
        total += read;
    }
    CloseHandle(file);
    return true;
#else
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    size_t total = 0;
    while (total < chunk.bytes) {
        ssize_t read = pread(fd,
                             dst + chunk.dst_offset + total,
                             chunk.bytes - total,
                             static_cast<off_t>(chunk.file_offset + total));
        if (read <= 0) {
            close(fd);
            return false;
        }
        total += static_cast<size_t>(read);
    }
    close(fd);
    return true;
#endif
}

}  // namespace

ThreadedFileReader::ThreadedFileReader(FileReadOptions options)
    : options_(options) {
    if (options_.worker_count == 0) {
        options_.worker_count = 1;
    }
    if (options_.chunk_bytes == 0) {
        options_.chunk_bytes = 8ull * 1024ull * 1024ull;
    }
}

bool ThreadedFileReader::read(const std::string& path, uint64_t offset, void* dst, size_t bytes) const {
    if (dst == nullptr) {
        return false;
    }
    if (bytes == 0) {
        return true;
    }

    const auto start = std::chrono::steady_clock::now();
    std::vector<ReadChunk> chunks;
    for (size_t pos = 0; pos < bytes; pos += options_.chunk_bytes) {
        const size_t chunk_bytes = std::min(options_.chunk_bytes, bytes - pos);
        chunks.push_back(ReadChunk{offset + pos, pos, chunk_bytes});
    }
    add_read_call(static_cast<uint64_t>(chunks.size()), static_cast<uint64_t>(bytes));

    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    const uint32_t worker_count = std::min<uint32_t>(options_.worker_count, static_cast<uint32_t>(chunks.size()));
    std::vector<std::thread> workers;
    char* out = static_cast<char*>(dst);
    for (uint32_t worker = 0; worker < std::max<uint32_t>(1, worker_count); ++worker) {
        workers.emplace_back([&, out]() {
            while (!failed.load()) {
                const size_t index = next.fetch_add(1);
                if (index >= chunks.size()) {
                    break;
                }
                if (!read_chunk(path, chunks[index], out)) {
                    failed = true;
                    break;
                }
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start).count();
    if (!failed.load()) {
        add_disk_read(static_cast<uint64_t>(bytes), ms);
        add_disk_read_wall(ms);
    } else {
        note_fallback(static_cast<uint64_t>(bytes));
    }
    return !failed.load();
}

}  // namespace loader
}  // namespace sd
