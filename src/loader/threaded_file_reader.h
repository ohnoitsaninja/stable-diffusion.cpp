#ifndef SD_THREADED_FILE_READER_H
#define SD_THREADED_FILE_READER_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace sd {
namespace loader {

struct FileReadOptions {
    uint32_t worker_count = 4;
    size_t chunk_bytes = 8ull * 1024ull * 1024ull;
};

class ThreadedFileReader {
public:
    explicit ThreadedFileReader(FileReadOptions options = {});

    bool read(const std::string& path, uint64_t offset, void* dst, size_t bytes) const;

private:
    FileReadOptions options_{};
};

}  // namespace loader
}  // namespace sd

#endif  // SD_THREADED_FILE_READER_H
