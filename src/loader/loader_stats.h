#ifndef SD_LOADER_STATS_H
#define SD_LOADER_STATS_H

#include <cstdint>

namespace sd {
namespace loader {

struct LoaderConfig {
    bool enable_threaded_loader = true;
    bool enable_pinned_staging = true;
    uint32_t read_threads = 4;
    uint64_t pin_budget_bytes = 1024ull * 1024ull * 1024ull;
    uint64_t ram_headroom_bytes = 2ull * 1024ull * 1024ull * 1024ull;
    uint64_t max_staging_bytes = 256ull * 1024ull * 1024ull;
    uint64_t min_tensor_bytes = 4ull * 1024ull * 1024ull;
};

struct LoaderStats {
    uint64_t disk_read_bytes = 0;
    uint64_t pinned_bytes_peak = 0;
    uint64_t h2d_bytes = 0;
    uint64_t fallback_bytes = 0;
    uint64_t read_call_count = 0;
    uint64_t read_chunk_count = 0;
    uint64_t read_chunk_bytes = 0;
    uint64_t tensor_count = 0;
    uint64_t cuda_host_register_count = 0;
    uint64_t cuda_host_unregister_count = 0;
    uint64_t cuda_stream_synchronize_count = 0;
    uint64_t cuda_device_synchronize_count = 0;
    double disk_read_ms = 0.0;
    double disk_read_wall_ms = 0.0;
    double h2d_ms = 0.0;
    double h2d_event_ms = 0.0;
    double total_model_load_ms = 0.0;
    double tensor_bookkeeping_ms = 0.0;
    double model_construction_ms = 0.0;
    double lora_patch_prep_ms = 0.0;
    uint32_t fallback_count = 0;
};

LoaderConfig default_config();
LoaderConfig get_config();
void set_config(const LoaderConfig& config);

LoaderStats get_stats();
void reset_stats();
void add_disk_read(uint64_t bytes, double ms);
void add_disk_read_wall(double ms);
void add_read_call(uint64_t chunks, uint64_t bytes);
void add_h2d(uint64_t bytes, double sync_ms, double event_ms);
void add_model_load_ms(double ms);
void add_tensor_bookkeeping(double ms);
void add_model_construction_ms(double ms);
void add_lora_patch_prep_ms(double ms);
void add_tensor_count(uint64_t tensors);
void note_pinned_bytes(uint64_t bytes);
void note_fallback(uint64_t bytes = 0);
void note_host_register();
void note_host_unregister();
void note_stream_synchronize();
void note_device_synchronize();

}  // namespace loader
}  // namespace sd

#endif  // SD_LOADER_STATS_H
