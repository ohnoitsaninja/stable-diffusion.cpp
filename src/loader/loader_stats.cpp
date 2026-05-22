#include "loader_stats.h"

#include <algorithm>
#include <mutex>

namespace sd {
namespace loader {

namespace {

std::mutex g_loader_mutex;
LoaderConfig g_loader_config = default_config();
LoaderStats g_loader_stats;

void add_fallback_reason_locked(LoaderFallbackReason reason) {
    switch (reason) {
        case LoaderFallbackReason::below_threshold:
            g_loader_stats.fallback_below_threshold_count += 1;
            break;
        case LoaderFallbackReason::host_destination:
            g_loader_stats.fallback_host_destination_count += 1;
            break;
        case LoaderFallbackReason::null_destination:
            g_loader_stats.fallback_null_destination_count += 1;
            break;
        case LoaderFallbackReason::zip_or_indirect:
            g_loader_stats.fallback_zip_or_indirect_count += 1;
            break;
        case LoaderFallbackReason::conversion_required:
            g_loader_stats.fallback_conversion_required_count += 1;
            break;
        case LoaderFallbackReason::type_mismatch:
            g_loader_stats.fallback_type_mismatch_count += 1;
            break;
        case LoaderFallbackReason::unsupported_backend:
            g_loader_stats.fallback_unsupported_backend_count += 1;
            break;
        case LoaderFallbackReason::arena_unavailable:
            g_loader_stats.fallback_arena_unavailable_count += 1;
            break;
        case LoaderFallbackReason::other:
        case LoaderFallbackReason::count:
            g_loader_stats.fallback_other_count += 1;
            break;
    }
}

}  // namespace

LoaderConfig default_config() {
    return LoaderConfig{};
}

LoaderConfig get_config() {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    return g_loader_config;
}

void set_config(const LoaderConfig& config) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_config = config;
    if (g_loader_config.read_threads == 0) {
        g_loader_config.read_threads = 1;
    }
    if (g_loader_config.min_tensor_bytes == 0) {
        g_loader_config.min_tensor_bytes = 1;
    }
}

LoaderStats get_stats() {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    return g_loader_stats;
}

void reset_stats() {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats = {};
}

void add_disk_read(uint64_t bytes, double ms) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.disk_read_bytes += bytes;
    g_loader_stats.disk_read_ms += ms;
}

void add_disk_read_wall(double ms) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.disk_read_wall_ms += ms;
}

void add_read_call(uint64_t chunks, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.read_call_count += 1;
    g_loader_stats.read_chunk_count += chunks;
    g_loader_stats.read_chunk_bytes += bytes;
}

void add_h2d(uint64_t bytes, double sync_ms, double event_ms) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.h2d_bytes += bytes;
    g_loader_stats.h2d_ms += sync_ms;
    g_loader_stats.h2d_event_ms += event_ms;
}

void add_model_load_ms(double ms) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.total_model_load_ms += ms;
}

void add_tensor_bookkeeping(double ms) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.tensor_bookkeeping_ms += ms;
}

void add_model_construction_ms(double ms) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.model_construction_ms += ms;
}

void add_lora_patch_prep_ms(double ms) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.lora_patch_prep_ms += ms;
}

void add_tensor_count(uint64_t tensors) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.tensor_count += tensors;
}

void note_fast_path(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.fast_path_tensor_count += 1;
    g_loader_stats.fast_path_bytes += bytes;
    g_loader_stats.tensor_count += 1;
}

void note_pinned_bytes(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.pinned_bytes_peak = std::max(g_loader_stats.pinned_bytes_peak, bytes);
}

void note_fallback(uint64_t bytes) {
    note_fallback(LoaderFallbackReason::other, bytes, bytes > 0);
}

void note_fallback(LoaderFallbackReason reason, uint64_t bytes, bool tensor_fallback) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.fallback_count += 1;
    add_fallback_reason_locked(reason);
    if (tensor_fallback) {
        g_loader_stats.fallback_tensor_count += 1;
        g_loader_stats.fallback_bytes += bytes;
        g_loader_stats.tensor_count += 1;
    }
}

void note_fallback_reason(LoaderFallbackReason reason) {
    note_fallback(reason, 0, false);
}

void note_dry_run_tensor() {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.dry_run_tensor_count += 1;
}

void note_host_register() {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.cuda_host_register_count += 1;
}

void note_host_unregister() {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.cuda_host_unregister_count += 1;
}

void note_stream_synchronize() {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.cuda_stream_synchronize_count += 1;
}

void note_device_synchronize() {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.cuda_device_synchronize_count += 1;
}

}  // namespace loader
}  // namespace sd
