#include "pinned_host_arena.h"

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <Windows.h>
#else
#include <cstdlib>
#endif

#if defined(SD_CUDA_THREADED_WEIGHT_LOADER) && defined(SD_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace sd {
namespace loader {

namespace {

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

PinnedHostArena::PinnedHostArena(const LoaderConfig& config)
    : config_(config) {
    if (config_.max_staging_bytes == 0) {
        config_.max_staging_bytes = default_config().max_staging_bytes;
    }
    if (config_.pin_budget_bytes > 0) {
        config_.max_staging_bytes = std::min(config_.max_staging_bytes, config_.pin_budget_bytes);
    }
    reserved_bytes_ = static_cast<size_t>(config_.max_staging_bytes);

#if defined(_WIN32)
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    page_size_ = static_cast<size_t>(info.dwPageSize);
    base_ = VirtualAlloc(nullptr, reserved_bytes_, MEM_RESERVE, PAGE_READWRITE);
#else
    page_size_ = 4096;
    base_ = std::malloc(reserved_bytes_);
    committed_bytes_ = base_ ? reserved_bytes_ : 0;
#endif

    if (base_ == nullptr) {
        note_fallback_reason(LoaderFallbackReason::arena_unavailable);
        reserved_bytes_ = 0;
        return;
    }
    (void)ensure_committed(reserved_bytes_);
}

PinnedHostArena::~PinnedHostArena() {
    release();
}

PinnedHostSpan PinnedHostArena::acquire(size_t bytes, size_t alignment) {
    if (base_ == nullptr || bytes == 0) {
        return {};
    }
    const size_t aligned_offset = align_up(used_bytes_, alignment);
    const size_t required = aligned_offset + bytes;
    if (required > reserved_bytes_) {
        return {};
    }
    if (!ensure_committed(required)) {
        return {};
    }
    used_bytes_ = required;
    note_pinned_bytes(static_cast<uint64_t>(committed_bytes_));
    return PinnedHostSpan{static_cast<char*>(base_) + aligned_offset, bytes, pinned_};
}

void PinnedHostArena::reset() {
    used_bytes_ = 0;
}

bool PinnedHostArena::ensure_committed(size_t bytes) {
    const size_t required = align_up(bytes, page_size_);
    if (required <= committed_bytes_) {
        return true;
    }
    const size_t old_committed = committed_bytes_;

#if defined(_WIN32)
    void* committed = VirtualAlloc(static_cast<char*>(base_) + committed_bytes_,
                                   required - committed_bytes_,
                                   MEM_COMMIT,
                                   PAGE_READWRITE);
    if (committed == nullptr) {
        note_fallback_reason(LoaderFallbackReason::arena_unavailable);
        return false;
    }
#endif

    committed_bytes_ = required;
    prefault(old_committed, committed_bytes_);
    return register_committed();
}

bool PinnedHostArena::register_committed() {
    if (!config_.enable_pinned_staging || committed_bytes_ == 0 || registered_bytes_ >= committed_bytes_) {
        return true;
    }

#if defined(SD_CUDA_THREADED_WEIGHT_LOADER) && defined(SD_USE_CUDA)
    unregister();
    note_host_register();
    cudaError_t err = cudaHostRegister(base_, committed_bytes_, cudaHostRegisterPortable);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        pinned_ = false;
        registered_bytes_ = 0;
        note_fallback_reason(LoaderFallbackReason::arena_unavailable);
        return false;
    }
    pinned_ = true;
    registered_bytes_ = committed_bytes_;
    return true;
#else
    pinned_ = false;
    registered_bytes_ = committed_bytes_;
    return true;
#endif
}

void PinnedHostArena::unregister() {
#if defined(SD_CUDA_THREADED_WEIGHT_LOADER) && defined(SD_USE_CUDA)
    if (pinned_ && base_ != nullptr && registered_bytes_ > 0) {
        note_host_unregister();
        cudaError_t err = cudaHostUnregister(base_);
        if (err != cudaSuccess) {
            (void)cudaGetLastError();
        }
    }
#endif
    pinned_ = false;
    registered_bytes_ = 0;
}

void PinnedHostArena::release() {
    unregister();
#if defined(_WIN32)
    if (base_ != nullptr) {
        VirtualFree(base_, 0, MEM_RELEASE);
    }
#else
    std::free(base_);
#endif
    base_ = nullptr;
    reserved_bytes_ = 0;
    committed_bytes_ = 0;
    used_bytes_ = 0;
}

void PinnedHostArena::prefault(size_t old_committed, size_t new_committed) {
    if (base_ == nullptr || new_committed <= old_committed) {
        return;
    }
    const uint32_t workers = std::max<uint32_t>(1, config_.read_threads);
    const size_t begin = align_up(old_committed, page_size_);
    const size_t bytes = new_committed - begin;
    if (bytes == 0) {
        return;
    }

    std::vector<std::thread> threads;
    const size_t pages = bytes / page_size_;
    const size_t pages_per_worker = std::max<size_t>(1, (pages + workers - 1) / workers);
    for (uint32_t worker = 0; worker < workers; ++worker) {
        const size_t first_page = worker * pages_per_worker;
        if (first_page >= pages) {
            break;
        }
        const size_t last_page = std::min(pages, first_page + pages_per_worker);
        threads.emplace_back([=]() {
            volatile char* ptr = static_cast<volatile char*>(base_) + begin + first_page * page_size_;
            for (size_t page = first_page; page < last_page; ++page) {
                *ptr = *ptr;
                ptr += page_size_;
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

}  // namespace loader
}  // namespace sd
