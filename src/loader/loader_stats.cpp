#include "loader_stats.h"

#include <algorithm>
#include <mutex>

namespace sd {
namespace loader {

namespace {

std::mutex g_loader_mutex;
LoaderConfig g_loader_config = default_config();
LoaderStats g_loader_stats;

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

void note_pinned_bytes(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.pinned_bytes_peak = std::max(g_loader_stats.pinned_bytes_peak, bytes);
}

void note_fallback(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    g_loader_stats.fallback_count += 1;
    g_loader_stats.fallback_bytes += bytes;
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
