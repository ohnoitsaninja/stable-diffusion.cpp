#ifndef SD_PINNED_HOST_ARENA_H
#define SD_PINNED_HOST_ARENA_H

#include <cstddef>
#include <cstdint>

#include "loader_stats.h"

namespace sd {
namespace loader {

struct PinnedHostSpan {
    void* data = nullptr;
    size_t size = 0;
    bool pinned = false;
};

class PinnedHostArena {
public:
    explicit PinnedHostArena(const LoaderConfig& config);
    ~PinnedHostArena();

    PinnedHostArena(const PinnedHostArena&) = delete;
    PinnedHostArena& operator=(const PinnedHostArena&) = delete;

    PinnedHostSpan acquire(size_t bytes, size_t alignment = 256);
    void reset();

    size_t committed_bytes() const { return committed_bytes_; }
    size_t used_bytes() const { return used_bytes_; }
    bool pinned() const { return pinned_; }

private:
    bool ensure_committed(size_t bytes);
    bool register_committed();
    void unregister();
    void release();
    void prefault(size_t old_committed, size_t new_committed);

    LoaderConfig config_{};
    void* base_ = nullptr;
    size_t reserved_bytes_ = 0;
    size_t committed_bytes_ = 0;
    size_t registered_bytes_ = 0;
    size_t used_bytes_ = 0;
    size_t page_size_ = 4096;
    bool pinned_ = false;
};

}  // namespace loader
}  // namespace sd

#endif  // SD_PINNED_HOST_ARENA_H
