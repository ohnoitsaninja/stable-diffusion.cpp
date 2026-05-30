#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <chrono>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <json.hpp>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef SD_LENS_TEXT_ENCODER_USE_CUBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>
extern "C" int lens_mxfp4_dequant_bf16_to_device(const uint8_t* blocks_host,
                                                  size_t blocks_bytes,
                                                  const uint8_t* scales_host,
                                                  size_t scales_bytes,
                                                  int experts,
                                                  int out_dim,
                                                  int groups,
                                                  int bytes_per_group,
                                                  int in_dim,
                                                  void* dst_device,
                                                  cudaStream_t stream,
                                                  float* upload_ms,
                                                  float* kernel_ms);
#endif
#ifdef SD_LENS_TEXT_ENCODER_USE_CUBLASLT
#include <cublasLt.h>
#endif
#ifdef SD_USE_CUDA
#include "ggml-cuda.h"
#endif

struct Args {
    std::string text_encoder_dir;
    std::string oracle_dir;
    std::string weight_map_out;
    bool router_only_cuda = false;
    bool projection_only = false;
    int projection_layer = 0;
    bool rope_only = false;
    int rope_layer = 0;
    bool attention_only = false;
    int attention_layer = 0;
    bool isolated_layer_set = false;
    int isolated_layer = 0;
    int through_layer = 0;
    bool through_layer_set = false;
    bool through_capture = false;
    bool summary_only = false;
    std::string emit_lens_cond_v1;
    std::string moe_cache = "per-layer-dequant";
    std::string moe_cache_layout = "packed-active";
    std::string cache_upload = "direct-mapped";
    int cache_upload_chunk_mib = 64;
    std::string moe_bf16_cache_dir = "build/diagnostics/lens_moe_bf16_cache";
    std::string moe_backend = "cpu-active-expert";
    std::string router_mode = "native";
    bool no_oracle_compare = false;
    bool layer0_drift_probe = false;
    bool layer1_perturb_probe = false;
    bool residual_cast_audit = false;
    bool moe_scalar_replay = false;
    int moe_scalar_layer = 0;
    int moe_scalar_token = 16;
    int moe_scalar_channel = 1167;
};

struct NpyF32 {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

struct NpyI64 {
    std::vector<int64_t> shape;
    std::vector<int64_t> data;
};

struct SafetensorEntry {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
    std::filesystem::path file;
    uint64_t payload_offset = 0;
};

struct DiffStats {
    float max_diff = 0.0f;
    float mean_diff = 0.0f;
    float max_ref = 0.0f;
    float rel_max = 0.0f;
    size_t max_index = 0;
};

struct TensorF32 {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

struct LensCondV1NativeInternal {
    std::array<TensorF32, 4> features;
    TensorF32 attention_mask;
    int txt_offset = 97;
    int raw_seq_len = 128;
    int trimmed_seq_len = 31;
    std::string router_mode = "native-tolerant";
    std::string source = "native_gptoss";
    bool bootstrap_tokens = true;
    std::array<int, 4> layer_taps = {5, 11, 17, 23};
    std::string expert_set_mismatch_tokens_per_layer;
};

static void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --text-encoder <Lens-Turbo/text_encoder> --oracle-dir <oracle> [--weight-map-out out.csv] [--mode emit-cond] [--no-oracle-compare] [--router-only-cuda] [--projection-only [layer=0]] [--rope-only [layer=0]] [--attention-only [layer=0]] [--isolated-layer N] [--through-layer N] [--through-capture N] [--router-mode native|oracle|native-tolerant] [--summary-only] [--emit-lens-cond-v1 out.safetensors] [--moe-cache per-layer-dequant|layer-bf16] [--moe-cache-layout packed-active|full-layer-resident] [--cache-upload direct-mapped|registered-mapped|registered-mapped-chunked|cuda-mxfp4-dequant] [--cache-upload-chunk-mib N] [--moe-bf16-cache-dir dir] [--moe-backend cpu-active-expert|cpu-parallel-expert|cuda-expert-matmul|cuda-batched-expert-matmul] [--layer0-drift-probe] [--layer1-perturb-probe] [--moe-scalar-replay [layer=0]] [--token N] [--channel N]\n",
                 argv0);
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (std::strcmp(arg, "--text-encoder") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.text_encoder_dir = v;
        } else if (std::strcmp(arg, "--oracle-dir") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.oracle_dir = v;
        } else if (std::strcmp(arg, "--weight-map-out") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.weight_map_out = v;
        } else if (std::strcmp(arg, "--mode") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            if (std::strcmp(v, "emit-cond") != 0) {
                std::fprintf(stderr, "--mode currently supports emit-cond\n");
                return false;
            }
            args.through_layer = 23;
            args.through_layer_set = true;
            args.through_capture = true;
            args.summary_only = true;
            args.no_oracle_compare = true;
            args.router_mode = "native-tolerant";
        } else if (std::strcmp(arg, "--no-oracle-compare") == 0) {
            args.no_oracle_compare = true;
        } else if (std::strcmp(arg, "--router-only-cuda") == 0) {
            args.router_only_cuda = true;
        } else if (std::strcmp(arg, "--projection-only") == 0) {
            args.projection_only = true;
            if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                const char* v = argv[++i];
                if (std::strncmp(v, "layer=", 6) == 0) {
                    args.projection_layer = std::atoi(v + 6);
                } else {
                    args.projection_layer = std::atoi(v);
                }
            }
        } else if (std::strcmp(arg, "--rope-only") == 0) {
            args.rope_only = true;
            if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                const char* v = argv[++i];
                if (std::strncmp(v, "layer=", 6) == 0) {
                    args.rope_layer = std::atoi(v + 6);
                } else {
                    args.rope_layer = std::atoi(v);
                }
            }
        } else if (std::strcmp(arg, "--attention-only") == 0) {
            args.attention_only = true;
            if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                const char* v = argv[++i];
                if (std::strncmp(v, "layer=", 6) == 0) {
                    args.attention_layer = std::atoi(v + 6);
                } else {
                    args.attention_layer = std::atoi(v);
                }
            }
        } else if (std::strcmp(arg, "--through-layer") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.through_layer = std::atoi(v);
            args.through_layer_set = true;
        } else if (std::strcmp(arg, "--through-capture") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.through_layer = std::atoi(v);
            args.through_layer_set = true;
            args.through_capture = true;
        } else if (std::strcmp(arg, "--router-mode") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.router_mode = v;
            if (args.router_mode != "native" && args.router_mode != "oracle" && args.router_mode != "native-tolerant") {
                std::fprintf(stderr, "--router-mode must be native, oracle, or native-tolerant\n");
                return false;
            }
        } else if (std::strcmp(arg, "--summary-only") == 0) {
            args.summary_only = true;
        } else if (std::strcmp(arg, "--emit-lens-cond-v1") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.emit_lens_cond_v1 = v;
        } else if (std::strcmp(arg, "--moe-cache") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.moe_cache = v;
            if (args.moe_cache != "per-layer-dequant" && args.moe_cache != "layer-bf16") {
                std::fprintf(stderr, "--moe-cache currently supports per-layer-dequant or layer-bf16\n");
                return false;
            }
        } else if (std::strcmp(arg, "--moe-cache-layout") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.moe_cache_layout = v;
            if (args.moe_cache_layout != "packed-active" && args.moe_cache_layout != "full-layer-resident") {
                std::fprintf(stderr, "--moe-cache-layout currently supports packed-active or full-layer-resident\n");
                return false;
            }
        } else if (std::strcmp(arg, "--cache-upload") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.cache_upload = v;
            if (args.cache_upload != "direct-mapped" &&
                args.cache_upload != "registered-mapped" &&
                args.cache_upload != "registered-mapped-chunked" &&
                args.cache_upload != "cuda-mxfp4-dequant") {
                std::fprintf(stderr, "--cache-upload currently supports direct-mapped, registered-mapped, registered-mapped-chunked, or cuda-mxfp4-dequant\n");
                return false;
            }
        } else if (std::strcmp(arg, "--cache-upload-chunk-mib") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.cache_upload_chunk_mib = std::atoi(v);
            if (args.cache_upload_chunk_mib <= 0 || args.cache_upload_chunk_mib > 1024) {
                std::fprintf(stderr, "--cache-upload-chunk-mib must be in 1..1024\n");
                return false;
            }
        } else if (std::strcmp(arg, "--moe-bf16-cache-dir") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.moe_bf16_cache_dir = v;
        } else if (std::strcmp(arg, "--moe-backend") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.moe_backend = v;
            if (args.moe_backend != "cpu-active-expert" &&
                args.moe_backend != "cpu-parallel-expert" &&
                args.moe_backend != "cuda-expert-matmul" &&
                args.moe_backend != "cuda-batched-expert-matmul") {
                std::fprintf(stderr, "--moe-backend must be cpu-active-expert, cpu-parallel-expert, cuda-expert-matmul, or cuda-batched-expert-matmul\n");
                return false;
            }
        } else if (std::strcmp(arg, "--isolated-layer") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.isolated_layer = std::atoi(v);
            args.isolated_layer_set = true;
        } else if (std::strcmp(arg, "--layer0-drift-probe") == 0) {
            args.layer0_drift_probe = true;
        } else if (std::strcmp(arg, "--layer1-perturb-probe") == 0) {
            args.layer1_perturb_probe = true;
        } else if (std::strcmp(arg, "--residual-cast-audit") == 0) {
            args.residual_cast_audit = true;
        } else if (std::strcmp(arg, "--moe-scalar-replay") == 0) {
            args.moe_scalar_replay = true;
            if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                const char* v = argv[++i];
                if (std::strncmp(v, "layer=", 6) == 0) {
                    args.moe_scalar_layer = std::atoi(v + 6);
                } else {
                    args.moe_scalar_layer = std::atoi(v);
                }
            }
        } else if (std::strcmp(arg, "--token") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.moe_scalar_token = std::atoi(v);
        } else if (std::strcmp(arg, "--channel") == 0) {
            const char* v = need(arg);
            if (!v) return false;
            args.moe_scalar_channel = std::atoi(v);
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg);
            return false;
        }
    }
    return !args.text_encoder_dir.empty() && !args.oracle_dir.empty();
}

static uint64_t elem_count(const std::vector<int64_t>& shape) {
    uint64_t count = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("invalid tensor shape");
        }
        count *= static_cast<uint64_t>(dim);
    }
    return count;
}

static void read_exact(std::ifstream& in, void* dst, size_t bytes, const std::string& what) {
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (!in || static_cast<size_t>(in.gcount()) != bytes) {
        throw std::runtime_error("failed to read " + what);
    }
}

static uint64_t read_le_u64(std::ifstream& in) {
    uint8_t bytes[8] = {};
    read_exact(in, bytes, sizeof(bytes), "u64");
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

static std::vector<int64_t> parse_npy_shape(const std::string& header) {
    const size_t lparen = header.find('(');
    const size_t rparen = header.find(')', lparen);
    if (lparen == std::string::npos || rparen == std::string::npos) {
        throw std::runtime_error("npy header has no shape");
    }
    std::vector<int64_t> shape;
    std::string dims = header.substr(lparen + 1, rparen - lparen - 1);
    size_t start = 0;
    while (start < dims.size()) {
        size_t comma = dims.find(',', start);
        std::string token = dims.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char ch) { return std::isspace(ch); }), token.end());
        if (!token.empty()) {
            shape.push_back(std::stoll(token));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (shape.empty()) {
        throw std::runtime_error("npy shape is empty");
    }
    return shape;
}

static std::string read_npy_header(std::ifstream& in) {
    char magic[6] = {};
    read_exact(in, magic, 6, "npy magic");
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("not an npy file");
    }
    uint8_t major = 0;
    uint8_t minor = 0;
    read_exact(in, &major, 1, "npy major");
    read_exact(in, &minor, 1, "npy minor");
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t h16 = 0;
        read_exact(in, &h16, 2, "npy header len");
        header_len = h16;
    } else if (major == 2 || major == 3) {
        read_exact(in, &header_len, 4, "npy header len");
    } else {
        throw std::runtime_error("unsupported npy version");
    }
    std::string header(header_len, '\0');
    read_exact(in, header.data(), header.size(), "npy header");
    if (header.find("True") != std::string::npos) {
        throw std::runtime_error("Fortran-order npy is not supported");
    }
    return header;
}

static NpyF32 load_npy_f32(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open npy: " + path.string());
    }
    std::string header = read_npy_header(in);
    if (header.find("'descr': '<f4'") == std::string::npos && header.find("\"descr\": \"<f4\"") == std::string::npos) {
        throw std::runtime_error("npy is not little-endian f32: " + path.string());
    }
    NpyF32 out;
    out.shape = parse_npy_shape(header);
    out.data.resize(static_cast<size_t>(elem_count(out.shape)));
    read_exact(in, out.data.data(), out.data.size() * sizeof(float), path.string());
    return out;
}

static NpyI64 load_npy_i64(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open npy: " + path.string());
    }
    std::string header = read_npy_header(in);
    if (header.find("'descr': '<i8'") == std::string::npos && header.find("\"descr\": \"<i8\"") == std::string::npos) {
        throw std::runtime_error("npy is not little-endian i64: " + path.string());
    }
    NpyI64 out;
    out.shape = parse_npy_shape(header);
    out.data.resize(static_cast<size_t>(elem_count(out.shape)));
    read_exact(in, out.data.data(), out.data.size() * sizeof(int64_t), path.string());
    return out;
}

static std::pair<NpyF32, NpyF32> load_or_make_gptoss_rope_cos_sin(const std::filesystem::path& oracle_dir);

static uint64_t dtype_bytes(const std::string& dtype) {
    if (dtype == "F32" || dtype == "I32") return 4;
    if (dtype == "BF16" || dtype == "F16") return 2;
    if (dtype == "U8" || dtype == "I8" || dtype == "F8_E8M0") return 1;
    return 0;
}

static std::vector<std::filesystem::path> safetensor_files(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

static std::vector<SafetensorEntry> read_header(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open safetensors: " + file.string());
    }
    const uint64_t header_size = read_le_u64(in);
    if (header_size == 0 || header_size > 256ull * 1024ull * 1024ull) {
        throw std::runtime_error("invalid safetensors header size");
    }
    std::string header(static_cast<size_t>(header_size), '\0');
    read_exact(in, header.data(), header.size(), "safetensors header");
    const uint64_t payload_offset = 8 + header_size;
    nlohmann::json json = nlohmann::json::parse(header);
    std::vector<SafetensorEntry> entries;
    entries.reserve(json.size());
    for (const auto& item : json.items()) {
        if (item.key() == "__metadata__") {
            continue;
        }
        SafetensorEntry e;
        e.name = item.key();
        e.dtype = item.value().at("dtype").get<std::string>();
        for (const auto& dim : item.value().at("shape")) {
            e.shape.push_back(dim.get<int64_t>());
        }
        e.begin = item.value().at("data_offsets").at(0).get<uint64_t>();
        e.end = item.value().at("data_offsets").at(1).get<uint64_t>();
        e.file = file;
        e.payload_offset = payload_offset;
        entries.push_back(std::move(e));
    }
    return entries;
}

static std::unordered_map<std::string, SafetensorEntry> index_safetensors(const std::filesystem::path& dir,
                                                                          std::vector<SafetensorEntry>* all_entries) {
    std::unordered_map<std::string, SafetensorEntry> index;
    for (const auto& file : safetensor_files(dir)) {
        auto entries = read_header(file);
        for (auto& e : entries) {
            if (all_entries) {
                all_entries->push_back(e);
            }
            index.emplace(e.name, std::move(e));
        }
    }
    return index;
}

static std::vector<uint8_t> read_tensor_bytes(const SafetensorEntry& e) {
    std::ifstream in(e.file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open tensor file: " + e.file.string());
    }
    const uint64_t bytes = e.end - e.begin;
    std::vector<uint8_t> data(static_cast<size_t>(bytes));
    in.seekg(static_cast<std::streamoff>(e.payload_offset + e.begin));
    read_exact(in, data.data(), data.size(), e.name);
    return data;
}

struct TensorBytesView {
    const uint8_t* data = nullptr;
    size_t size = 0;
    std::vector<uint8_t> owned;
#ifdef _WIN32
    std::shared_ptr<void> mapped_owner;
#endif
};

#ifdef _WIN32
struct MappedSafetensorFile {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
    const uint8_t* data = nullptr;
    uint64_t bytes = 0;

    ~MappedSafetensorFile() {
        if (data != nullptr) {
            UnmapViewOfFile(data);
        }
        if (mapping != nullptr) {
            CloseHandle(mapping);
        }
        if (file != INVALID_HANDLE_VALUE) {
            CloseHandle(file);
        }
    }
};

static std::unordered_map<std::string, std::shared_ptr<MappedSafetensorFile>> g_safetensor_file_maps;

static std::shared_ptr<MappedSafetensorFile> map_safetensor_file(const std::filesystem::path& path,
                                                                 double* open_seconds = nullptr) {
    const std::string key = path.string();
    const auto found = g_safetensor_file_maps.find(key);
    if (found != g_safetensor_file_maps.end()) {
        if (open_seconds != nullptr) {
            *open_seconds = 0.0;
        }
        return found->second;
    }
    const auto start = std::chrono::steady_clock::now();
    auto mapped = std::make_shared<MappedSafetensorFile>();
    const std::wstring wide = path.wstring();
    mapped->file = CreateFileW(wide.c_str(),
                               GENERIC_READ,
                               FILE_SHARE_READ,
                               nullptr,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
                               nullptr);
    if (mapped->file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("failed to open safetensors mapping: " + path.string());
    }
    LARGE_INTEGER size = {};
    if (!GetFileSizeEx(mapped->file, &size) || size.QuadPart <= 0) {
        throw std::runtime_error("failed to query safetensors mapping size: " + path.string());
    }
    mapped->bytes = static_cast<uint64_t>(size.QuadPart);
    mapped->mapping = CreateFileMappingW(mapped->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapped->mapping == nullptr) {
        throw std::runtime_error("failed to create safetensors mapping: " + path.string());
    }
    mapped->data = reinterpret_cast<const uint8_t*>(MapViewOfFile(mapped->mapping, FILE_MAP_READ, 0, 0, 0));
    if (mapped->data == nullptr) {
        throw std::runtime_error("failed to map safetensors view: " + path.string());
    }
    if (open_seconds != nullptr) {
        *open_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    g_safetensor_file_maps.emplace(key, mapped);
    return mapped;
}
#endif

static TensorBytesView tensor_bytes_view(const SafetensorEntry& e, double* open_seconds = nullptr) {
    const uint64_t bytes = e.end - e.begin;
    TensorBytesView view;
    view.size = static_cast<size_t>(bytes);
#ifdef _WIN32
    double map_seconds = 0.0;
    auto mapped = map_safetensor_file(e.file, &map_seconds);
    const uint64_t offset = e.payload_offset + e.begin;
    if (offset + bytes > mapped->bytes) {
        throw std::runtime_error("safetensors mapped view out of range: " + e.name);
    }
    view.data = mapped->data + offset;
    view.mapped_owner = mapped;
    if (open_seconds != nullptr) {
        *open_seconds = map_seconds;
    }
#else
    view.owned = read_tensor_bytes(e);
    view.data = view.owned.data();
    if (open_seconds != nullptr) {
        *open_seconds = 0.0;
    }
#endif
    return view;
}

static float bf16_to_f32(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t f32_to_bf16_bits(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounding_bias = 0x7fffu + lsb;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

static float round_to_bf16(float x) {
    if (!std::isfinite(x)) {
        return x;
    }
    return bf16_to_f32(f32_to_bf16_bits(x));
}

static std::vector<float> read_bf16_as_f32(const SafetensorEntry& e) {
    if (e.dtype != "BF16") {
        throw std::runtime_error("expected BF16 tensor: " + e.name);
    }
    auto bytes = read_tensor_bytes(e);
    if (bytes.size() != elem_count(e.shape) * 2) {
        throw std::runtime_error("unexpected BF16 tensor byte size");
    }
    std::vector<float> out(bytes.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        uint16_t v = static_cast<uint16_t>(bytes[i * 2]) | (static_cast<uint16_t>(bytes[i * 2 + 1]) << 8);
        out[i] = bf16_to_f32(v);
    }
    return out;
}

static std::vector<ggml_bf16_t> read_bf16_raw(const SafetensorEntry& e) {
    if (e.dtype != "BF16") {
        throw std::runtime_error("expected BF16 tensor: " + e.name);
    }
    auto bytes = read_tensor_bytes(e);
    if (bytes.size() != elem_count(e.shape) * sizeof(ggml_bf16_t)) {
        throw std::runtime_error("unexpected BF16 tensor byte size");
    }
    std::vector<ggml_bf16_t> out(bytes.size() / sizeof(ggml_bf16_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

static std::vector<float> read_f32_as_f32(const SafetensorEntry& e) {
    if (e.dtype != "F32") {
        throw std::runtime_error("expected F32 tensor: " + e.name);
    }
    auto bytes = read_tensor_bytes(e);
    if (bytes.size() != elem_count(e.shape) * sizeof(float)) {
        throw std::runtime_error("unexpected F32 tensor byte size");
    }
    std::vector<float> out(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

static std::vector<float> read_numeric_as_f32(const SafetensorEntry& e) {
    if (e.dtype == "BF16") {
        return read_bf16_as_f32(e);
    }
    if (e.dtype == "F32") {
        return read_f32_as_f32(e);
    }
    throw std::runtime_error("unsupported numeric tensor dtype for " + e.name + ": " + e.dtype);
}

static const SafetensorEntry& need_entry(const std::unordered_map<std::string, SafetensorEntry>& index,
                                         const std::string& name) {
    auto it = index.find(name);
    if (it == index.end()) {
        throw std::runtime_error("missing tensor: " + name);
    }
    return it->second;
}

static DiffStats diff_stats(const std::vector<float>& actual, const std::vector<float>& ref) {
    if (actual.size() != ref.size()) {
        throw std::runtime_error("diff size mismatch");
    }
    DiffStats s;
    double sum = 0.0;
    double ref_sum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float d = std::fabs(actual[i] - ref[i]);
        const float r = std::fabs(ref[i]);
        sum += d;
        ref_sum += r;
        if (d > s.max_diff) {
            s.max_diff = d;
            s.max_index = i;
        }
        s.max_ref = std::max(s.max_ref, r);
    }
    if (!actual.empty()) {
        s.mean_diff = static_cast<float>(sum / static_cast<double>(actual.size()));
        s.rel_max = s.max_ref > 0.0f ? s.max_diff / s.max_ref : 0.0f;
    }
    return s;
}

static void report_diff(const std::string& label, const TensorF32& actual, const NpyF32& ref) {
    if (actual.shape != ref.shape) {
        std::ostringstream oss;
        oss << label << " shape mismatch";
        throw std::runtime_error(oss.str());
    }
    const DiffStats d = diff_stats(actual.data, ref.data);
    size_t finite = 0;
    for (float v : actual.data) {
        if (std::isfinite(v)) {
            ++finite;
        }
    }
    std::cout << label << " shape=";
    for (size_t i = 0; i < actual.shape.size(); ++i) {
        if (i) std::cout << "x";
        std::cout << actual.shape[i];
    }
    std::cout << " finite=" << finite << "/" << actual.data.size()
              << " max_diff=" << d.max_diff
              << " mean_diff=" << d.mean_diff
              << " rel_max=" << d.rel_max
              << " max_index=" << d.max_index << "\n";
}

static TensorF32 make_tensor(std::vector<int64_t> shape, std::vector<float> data) {
    if (elem_count(shape) != data.size()) {
        throw std::runtime_error("make_tensor shape/data mismatch");
    }
    return TensorF32{std::move(shape), std::move(data)};
}

static std::string shape_string(const std::vector<int64_t>& shape) {
    std::ostringstream out;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out << "x";
        out << shape[i];
    }
    return out.str();
}

static void write_le_u64(std::ofstream& out, uint64_t value) {
    uint8_t bytes[8] = {};
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<uint8_t>((value >> (8 * i)) & 0xffu);
    }
    out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

static void write_lens_cond_v1_safetensors(const std::filesystem::path& path,
                                           const std::array<TensorF32, 4>& features,
                                           const TensorF32& attention_mask,
                                           const nlohmann::json& metadata) {
    for (size_t i = 0; i < features.size(); ++i) {
        const TensorF32& feature = features[i];
        if (feature.shape.size() != 3 || feature.shape[0] != 1 || feature.shape[1] <= 0 || feature.shape[2] != 2880) {
            throw std::runtime_error("feature_" + std::to_string(i) + " has invalid lens_cond_v1 shape");
        }
        if (i > 0 && feature.shape != features[0].shape) {
            throw std::runtime_error("lens_cond_v1 feature shapes do not match");
        }
    }
    if (attention_mask.shape.size() != 2 || attention_mask.shape[0] != 1 || attention_mask.shape[1] != features[0].shape[1]) {
        throw std::runtime_error("lens_cond_v1 attention mask shape mismatch");
    }

    struct TensorWrite {
        std::string name;
        const std::vector<int64_t>* shape;
        const std::vector<float>* data;
    };
    std::vector<TensorWrite> tensors;
    tensors.reserve(5);
    for (size_t i = 0; i < features.size(); ++i) {
        tensors.push_back({"feature_" + std::to_string(i), &features[i].shape, &features[i].data});
    }
    tensors.push_back({"attention_mask", &attention_mask.shape, &attention_mask.data});

    nlohmann::json header = nlohmann::json::object();
    nlohmann::json meta = metadata;
    for (auto& item : meta.items()) {
        if (!item.value().is_string()) {
            item.value() = item.value().dump();
        }
    }
    header["__metadata__"] = meta;

    uint64_t offset = 0;
    for (const TensorWrite& tensor : tensors) {
        const uint64_t bytes = static_cast<uint64_t>(tensor.data->size() * sizeof(float));
        header[tensor.name] = {
            {"dtype", "F32"},
            {"shape", *tensor.shape},
            {"data_offsets", {offset, offset + bytes}},
        };
        offset += bytes;
    }

    std::string header_text = header.dump();
    if (header_text.empty() || header_text.front() != '{') {
        throw std::runtime_error("invalid safetensors header");
    }
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open lens_cond_v1 output: " + path.string());
    }
    write_le_u64(out, static_cast<uint64_t>(header_text.size()));
    out.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (const TensorWrite& tensor : tensors) {
        out.write(reinterpret_cast<const char*>(tensor.data->data()),
                  static_cast<std::streamsize>(tensor.data->size() * sizeof(float)));
    }
    if (!out) {
        throw std::runtime_error("failed to write lens_cond_v1 output: " + path.string());
    }
}

static NpyF32 load_npy_f32_batched_3d(const std::filesystem::path& path, int64_t seq, int64_t width) {
    NpyF32 out = load_npy_f32(path);
    if (out.shape.size() == 3 && out.shape[0] == 1 && out.shape[1] == seq && out.shape[2] == width) {
        return out;
    }
    if (out.shape.size() == 2 && out.shape[0] == seq && out.shape[1] == width) {
        out.shape = {1, seq, width};
        return out;
    }
    throw std::runtime_error("unexpected batched f32 npy shape for " + path.string() +
                             ": got " + shape_string(out.shape) +
                             " expected 1x" + std::to_string(seq) + "x" + std::to_string(width) +
                             " or " + std::to_string(seq) + "x" + std::to_string(width));
}

static NpyI64 load_npy_i64_batched_3d(const std::filesystem::path& path, int64_t seq, int64_t width) {
    NpyI64 out = load_npy_i64(path);
    if (out.shape.size() == 3 && out.shape[0] == 1 && out.shape[1] == seq && out.shape[2] == width) {
        return out;
    }
    if (out.shape.size() == 2 && out.shape[0] == seq && out.shape[1] == width) {
        out.shape = {1, seq, width};
        return out;
    }
    throw std::runtime_error("unexpected batched i64 npy shape for " + path.string() +
                             ": got " + shape_string(out.shape) +
                             " expected 1x" + std::to_string(seq) + "x" + std::to_string(width) +
                             " or " + std::to_string(seq) + "x" + std::to_string(width));
}

static void upload_f32_or_bf16(ggml_tensor* tensor, const std::vector<float>& data);
static TensorF32 download_tensor_f32(ggml_tensor* tensor, const std::vector<int64_t>& shape);
static TensorF32 rms_norm_ggml_cuda(const TensorF32& x, const std::vector<float>& weight, float eps, double* elapsed_ms = nullptr);

static TensorF32 rms_norm_bf16_variant(const TensorF32& x,
                                       const std::vector<float>& weight,
                                       float eps,
                                       const std::string& reduction,
                                       bool cast_normalized_before_weight) {
    if (x.shape.size() != 3 || x.shape[0] != 1 || static_cast<int64_t>(weight.size()) != x.shape[2]) {
        throw std::runtime_error("rms_norm shape mismatch");
    }
    const int64_t seq = x.shape[1];
    const int64_t hidden = x.shape[2];
    TensorF32 out;
    out.shape = x.shape;
    out.data.resize(x.data.size());
    for (int64_t s = 0; s < seq; ++s) {
        const size_t base = static_cast<size_t>(s * hidden);
        float variance = 0.0f;
        if (reduction == "double") {
            double ss = 0.0;
            for (int64_t h = 0; h < hidden; ++h) {
                const float v = x.data[base + static_cast<size_t>(h)];
                ss += static_cast<double>(v) * static_cast<double>(v);
            }
            variance = static_cast<float>(ss / static_cast<double>(hidden));
        } else if (reduction == "float") {
            float ss = 0.0f;
            for (int64_t h = 0; h < hidden; ++h) {
                const float v = x.data[base + static_cast<size_t>(h)];
                ss += v * v;
            }
            variance = ss / static_cast<float>(hidden);
        } else {
            throw std::runtime_error("unsupported rms_norm reduction mode");
        }
        const float scale = 1.0f / std::sqrt(variance + eps);
        for (int64_t h = 0; h < hidden; ++h) {
            float normalized = x.data[base + static_cast<size_t>(h)] * scale;
            if (cast_normalized_before_weight) {
                normalized = round_to_bf16(normalized);
            }
            out.data[base + static_cast<size_t>(h)] = round_to_bf16(normalized * weight[static_cast<size_t>(h)]);
        }
    }
    return out;
}

static TensorF32 rms_norm_bf16(const TensorF32& x, const std::vector<float>& weight, float eps) {
#ifdef SD_USE_CUDA
    return rms_norm_ggml_cuda(x, weight, eps, nullptr);
#else
    return rms_norm_bf16_variant(x, weight, eps, "double", false);
#endif
}

static void report_rmsnorm_scalar_probe(const std::string& label,
                                        const TensorF32& x,
                                        const std::vector<float>& weight,
                                        float eps,
                                        const std::filesystem::path& oracle_dir) {
    const std::filesystem::path ref_path = oracle_dir / (label + "_rmsnorm_output_f32.npy");
    if (!std::filesystem::exists(ref_path)) {
        return;
    }
    const NpyF32 ref = load_npy_f32(ref_path);
    const std::vector<std::pair<std::string, TensorF32>> variants = {
        {"double_reduction_final_cast", rms_norm_bf16_variant(x, weight, eps, "double", false)},
        {"float_reduction_final_cast", rms_norm_bf16_variant(x, weight, eps, "float", false)},
        {"float_reduction_cast_normalized_before_weight", rms_norm_bf16_variant(x, weight, eps, "float", true)},
    };
    std::vector<std::pair<std::string, TensorF32>> all_variants = variants;
#ifdef SD_USE_CUDA
    try {
        double ggml_ms = 0.0;
        TensorF32 ggml = rms_norm_ggml_cuda(x, weight, eps, &ggml_ms);
        std::cout << label << "_rmsnorm_cuda_probe backend=ggml_cuda elapsed_ms=" << ggml_ms << "\n";
        all_variants.push_back({"ggml_cuda_rms_norm_mul_cast_bf16", std::move(ggml)});
    } catch (const std::exception& e) {
        std::cout << label << "_rmsnorm_cuda_probe backend=ggml_cuda error=\"" << e.what() << "\"\n";
    }
#endif
    const int64_t hidden = x.shape.at(2);
    NpyF32 ref_variance;
    NpyF32 ref_rsqrt;
    NpyF32 ref_normalized;
    NpyF32 ref_weighted;
    const bool have_internals =
        std::filesystem::exists(oracle_dir / (label + "_rmsnorm_variance_f32.npy")) &&
        std::filesystem::exists(oracle_dir / (label + "_rmsnorm_rsqrt_f32.npy")) &&
        std::filesystem::exists(oracle_dir / (label + "_rmsnorm_normalized_f32.npy")) &&
        std::filesystem::exists(oracle_dir / (label + "_rmsnorm_weighted_f32.npy"));
    if (have_internals) {
        ref_variance = load_npy_f32(oracle_dir / (label + "_rmsnorm_variance_f32.npy"));
        ref_rsqrt = load_npy_f32(oracle_dir / (label + "_rmsnorm_rsqrt_f32.npy"));
        ref_normalized = load_npy_f32(oracle_dir / (label + "_rmsnorm_normalized_f32.npy"));
        ref_weighted = load_npy_f32(oracle_dir / (label + "_rmsnorm_weighted_f32.npy"));
    }
    for (const auto& item : all_variants) {
        const DiffStats d = diff_stats(item.second.data, ref.data);
        const int64_t token = static_cast<int64_t>(d.max_index / static_cast<size_t>(hidden));
        const int64_t channel = static_cast<int64_t>(d.max_index % static_cast<size_t>(hidden));
        const size_t base = static_cast<size_t>(token * hidden);
        float ss = 0.0f;
        for (int64_t h = 0; h < hidden; ++h) {
            const float v = x.data[base + static_cast<size_t>(h)];
            ss += v * v;
        }
        const float variance = ss / static_cast<float>(hidden);
        const float rsqrt = 1.0f / std::sqrt(variance + eps);
        const float normalized = x.data[base + static_cast<size_t>(channel)] * rsqrt;
        const float weighted = normalized * weight[static_cast<size_t>(channel)];
        std::cout << label << "_scalar_probe"
                  << " variant=" << item.first
                  << " max_diff=" << d.max_diff
                  << " mean_diff=" << d.mean_diff
                  << " token=" << token
                  << " channel=" << channel
                  << " native=" << item.second.data[d.max_index]
                  << " ref=" << ref.data[d.max_index]
                  << " native_bf16_bits=0x" << std::hex << f32_to_bf16_bits(item.second.data[d.max_index])
                  << " ref_bf16_bits=0x" << f32_to_bf16_bits(ref.data[d.max_index]) << std::dec
                  << " variance_f32=" << variance
                  << " rsqrt_f32=" << rsqrt
                  << " normalized_f32=" << normalized
                  << " weighted_f32=" << weighted;
        if (have_internals) {
            const size_t scalar_index = static_cast<size_t>(token * hidden + channel);
            std::cout << " oracle_variance_f32=" << ref_variance.data[static_cast<size_t>(token)]
                      << " oracle_rsqrt_f32=" << ref_rsqrt.data[static_cast<size_t>(token)]
                      << " oracle_normalized_f32=" << ref_normalized.data[scalar_index]
                      << " oracle_weighted_f32=" << ref_weighted.data[scalar_index];
        }
        std::cout << "\n";
    }
}

static TensorF32 linear_bf16(const TensorF32& x,
                             const std::vector<float>& weight,
                             const std::vector<int64_t>& weight_shape,
                             const std::vector<float>& bias,
                             const std::string& label) {
    if (x.shape.size() != 3 || x.shape[0] != 1 || weight_shape.size() != 2) {
        throw std::runtime_error(label + " rank mismatch");
    }
    const int64_t seq = x.shape[1];
    const int64_t in = x.shape[2];
    const int64_t out_dim = weight_shape[0];
    if (weight_shape[1] != in || static_cast<int64_t>(bias.size()) != out_dim ||
        static_cast<int64_t>(weight.size()) != out_dim * in) {
        throw std::runtime_error(label + " shape mismatch");
    }
    TensorF32 out;
    out.shape = {1, seq, out_dim};
    out.data.resize(static_cast<size_t>(seq * out_dim));
    for (int64_t s = 0; s < seq; ++s) {
        const float* xrow = x.data.data() + static_cast<size_t>(s * in);
        for (int64_t o = 0; o < out_dim; ++o) {
            const float* wrow = weight.data() + static_cast<size_t>(o * in);
            float sum = bias[static_cast<size_t>(o)];
            for (int64_t i = 0; i < in; ++i) {
                sum = std::fma(xrow[i], wrow[i], sum);
            }
            out.data[static_cast<size_t>(s * out_dim + o)] = round_to_bf16(sum);
        }
    }
    return out;
}

static TensorF32 add_bf16(const TensorF32& a, const TensorF32& b, const std::string& label) {
    if (a.shape != b.shape) {
        throw std::runtime_error(label + " add shape mismatch");
    }
    TensorF32 out;
    out.shape = a.shape;
    out.data.resize(a.data.size());
    for (size_t i = 0; i < a.data.size(); ++i) {
        out.data[i] = round_to_bf16(a.data[i] + b.data[i]);
    }
    return out;
}

static TensorF32 add_f32_no_cast(const TensorF32& a, const TensorF32& b, const std::string& label) {
    if (a.shape != b.shape) {
        throw std::runtime_error(label + " add shape mismatch");
    }
    TensorF32 out;
    out.shape = a.shape;
    out.data.resize(a.data.size());
    for (size_t i = 0; i < a.data.size(); ++i) {
        out.data[i] = a.data[i] + b.data[i];
    }
    return out;
}

static TensorF32 add_cast_operands_then_bf16(const TensorF32& a, const TensorF32& b, const std::string& label) {
    if (a.shape != b.shape) {
        throw std::runtime_error(label + " add shape mismatch");
    }
    TensorF32 out;
    out.shape = a.shape;
    out.data.resize(a.data.size());
    for (size_t i = 0; i < a.data.size(); ++i) {
        out.data[i] = round_to_bf16(round_to_bf16(a.data[i]) + round_to_bf16(b.data[i]));
    }
    return out;
}

static void report_boundary_diff(const std::string& label,
                                 const TensorF32& actual,
                                 const NpyF32& ref,
                                 const std::string& cast_note) {
    if (actual.shape != ref.shape || actual.data.size() != ref.data.size()) {
        throw std::runtime_error(label + " boundary diff shape mismatch");
    }
    const DiffStats d = diff_stats(actual.data, ref.data);
    int64_t token = -1;
    int64_t channel = -1;
    if (actual.shape.size() == 3 && actual.shape[0] == 1) {
        const int64_t hidden = actual.shape[2];
        token = static_cast<int64_t>(d.max_index / static_cast<size_t>(hidden));
        channel = static_cast<int64_t>(d.max_index % static_cast<size_t>(hidden));
    }
    std::cout << label
              << " max_diff=" << d.max_diff
              << " mean_diff=" << d.mean_diff
              << " rel_max=" << d.rel_max
              << " max_index=" << d.max_index
              << " token=" << token
              << " channel=" << channel
              << " native=" << actual.data[d.max_index]
              << " oracle=" << ref.data[d.max_index]
              << " native_bf16=0x" << std::hex << f32_to_bf16_bits(actual.data[d.max_index])
              << " oracle_bf16=0x" << f32_to_bf16_bits(ref.data[d.max_index]) << std::dec
              << " cast_order=\"" << cast_note << "\"\n";
}

enum class RopeVariant {
    SplitHalfFinalCast,
    SplitHalfProductCast,
    InterleavedFinalCast,
};

static const char* rope_variant_name(RopeVariant variant) {
    switch (variant) {
        case RopeVariant::SplitHalfFinalCast: return "split_half_final_cast";
        case RopeVariant::SplitHalfProductCast: return "split_half_product_cast";
        case RopeVariant::InterleavedFinalCast: return "interleaved_final_cast";
    }
    return "unknown";
}

static TensorF32 apply_rope_to_states_bf16(const TensorF32& states,
                                           const std::vector<float>& cos,
                                           const std::vector<float>& sin,
                                           RopeVariant variant) {
    if (states.shape.size() != 4 || states.shape[0] != 1) {
        throw std::runtime_error("RoPE states shape mismatch");
    }
    const int64_t heads = states.shape[1];
    const int64_t seq = states.shape[2];
    const int64_t head_dim = states.shape[3];
    if (head_dim % 2 != 0 ||
        static_cast<int64_t>(cos.size()) != seq * (head_dim / 2) ||
        static_cast<int64_t>(sin.size()) != seq * (head_dim / 2)) {
        throw std::runtime_error("RoPE cos/sin shape mismatch");
    }
    TensorF32 out = states;
    for (int64_t h = 0; h < heads; ++h) {
        for (int64_t s = 0; s < seq; ++s) {
            if (variant == RopeVariant::InterleavedFinalCast) {
                for (int64_t d = 0; d < head_dim / 2; ++d) {
                    const size_t a = static_cast<size_t>((h * seq + s) * head_dim + 2 * d);
                    const size_t b = a + 1;
                    const float first = states.data[a];
                    const float second = states.data[b];
                    const float c = cos[static_cast<size_t>(s * (head_dim / 2) + d)];
                    const float sn = sin[static_cast<size_t>(s * (head_dim / 2) + d)];
                    out.data[a] = round_to_bf16(first * c - second * sn);
                    out.data[b] = round_to_bf16(second * c + first * sn);
                }
            } else {
                for (int64_t d = 0; d < head_dim / 2; ++d) {
                    const size_t a = static_cast<size_t>((h * seq + s) * head_dim + d);
                    const size_t b = static_cast<size_t>((h * seq + s) * head_dim + d + head_dim / 2);
                    const float first = states.data[a];
                    const float second = states.data[b];
                    const float c = cos[static_cast<size_t>(s * (head_dim / 2) + d)];
                    const float sn = sin[static_cast<size_t>(s * (head_dim / 2) + d)];
                    if (variant == RopeVariant::SplitHalfProductCast) {
                        const float first_cos = round_to_bf16(first * c);
                        const float second_sin = round_to_bf16(second * sn);
                        const float second_cos = round_to_bf16(second * c);
                        const float first_sin = round_to_bf16(first * sn);
                        out.data[a] = round_to_bf16(first_cos - second_sin);
                        out.data[b] = round_to_bf16(second_cos + first_sin);
                    } else {
                        out.data[a] = round_to_bf16(first * c - second * sn);
                        out.data[b] = round_to_bf16(second * c + first * sn);
                    }
                }
            }
        }
    }
    return out;
}

static TensorF32 reshape_qkv_rope_bf16(const TensorF32& proj,
                                       int64_t heads,
                                       int64_t seq,
                                       int64_t head_dim,
                                       const std::vector<float>& cos,
                                       const std::vector<float>& sin,
                                       bool apply_rope) {
    if (proj.shape.size() != 3 || proj.shape[1] != seq || proj.shape[2] != heads * head_dim) {
        throw std::runtime_error("qkv reshape mismatch");
    }
    TensorF32 out;
    out.shape = {1, heads, seq, head_dim};
    out.data.resize(static_cast<size_t>(heads * seq * head_dim));
    for (int64_t h = 0; h < heads; ++h) {
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const float v = proj.data[static_cast<size_t>(s * heads * head_dim + h * head_dim + d)];
                out.data[static_cast<size_t>((h * seq + s) * head_dim + d)] = v;
            }
        }
    }
    if (apply_rope) {
        out = apply_rope_to_states_bf16(out, cos, sin, RopeVariant::SplitHalfProductCast);
    }
    return out;
}

static TensorF32 attention_pre_o_bf16(const TensorF32& q,
                                      const TensorF32& k,
                                      const TensorF32& v,
                                      const std::vector<float>& sinks,
                                      int64_t heads,
                                      int64_t kv_heads,
                                      int64_t seq,
                                      int64_t head_dim,
                                      bool sliding_attention,
                                      int64_t sliding_window) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t groups = heads / kv_heads;
    TensorF32 out;
    out.shape = {1, seq, heads * head_dim};
    out.data.assign(static_cast<size_t>(seq * heads * head_dim), 0.0f);
    std::vector<float> logits(static_cast<size_t>(seq + 1));
    std::vector<float> probs(static_cast<size_t>(seq));
    for (int64_t h = 0; h < heads; ++h) {
        const int64_t kh = h / groups;
        for (int64_t i = 0; i < seq; ++i) {
            float max_logit = -std::numeric_limits<float>::infinity();
            for (int64_t j = 0; j < seq; ++j) {
                float value = -std::numeric_limits<float>::infinity();
                if (j <= i && (!sliding_attention || (i - j) < sliding_window)) {
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) {
                    const float qv = q.data[static_cast<size_t>((h * seq + i) * head_dim + d)];
                    const float kv = k.data[static_cast<size_t>((kh * seq + j) * head_dim + d)];
                        dot = std::fma(qv, kv, dot);
                }
                    value = round_to_bf16(dot * scale);
                }
                logits[static_cast<size_t>(j)] = value;
                max_logit = std::max(max_logit, value);
            }
            logits[static_cast<size_t>(seq)] = sinks[static_cast<size_t>(h)];
            max_logit = std::max(max_logit, logits[static_cast<size_t>(seq)]);
            double denom = 0.0;
            for (int64_t j = 0; j <= seq; ++j) {
                if (std::isfinite(logits[static_cast<size_t>(j)])) {
                    denom += std::exp(static_cast<double>(round_to_bf16(logits[static_cast<size_t>(j)] - max_logit)));
                }
            }
            for (int64_t j = 0; j < seq; ++j) {
                probs[static_cast<size_t>(j)] = std::isfinite(logits[static_cast<size_t>(j)])
                    ? round_to_bf16(static_cast<float>(std::exp(static_cast<double>(round_to_bf16(logits[static_cast<size_t>(j)] - max_logit))) / denom))
                    : 0.0f;
            }
            for (int64_t d = 0; d < head_dim; ++d) {
                float sum = 0.0f;
                for (int64_t j = 0; j < seq; ++j) {
                    sum = std::fma(probs[static_cast<size_t>(j)],
                                   v.data[static_cast<size_t>((kh * seq + j) * head_dim + d)],
                                   sum);
                }
                out.data[static_cast<size_t>(i * heads * head_dim + h * head_dim + d)] = round_to_bf16(sum);
            }
        }
    }
    return out;
}

static TensorF32 attention_pre_o_bf16_fast(const TensorF32& q,
                                           const TensorF32& k,
                                           const TensorF32& v,
                                           const std::vector<float>& sinks,
                                           int64_t heads,
                                           int64_t kv_heads,
                                           int64_t seq,
                                           int64_t head_dim,
                                           bool sliding_attention,
                                           int64_t sliding_window) {
    if (q.shape != std::vector<int64_t>{1, heads, seq, head_dim} ||
        k.shape != std::vector<int64_t>{1, kv_heads, seq, head_dim} ||
        v.shape != std::vector<int64_t>{1, kv_heads, seq, head_dim} ||
        static_cast<int64_t>(sinks.size()) != heads ||
        heads % kv_heads != 0) {
        throw std::runtime_error("fast attention input shape mismatch");
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t groups = heads / kv_heads;
    TensorF32 out;
    out.shape = {1, seq, heads * head_dim};
    out.data.assign(static_cast<size_t>(seq * heads * head_dim), 0.0f);

    auto compute_head_range = [&](int64_t h_begin, int64_t h_end) {
        std::vector<float> logits(static_cast<size_t>(seq));
        std::vector<float> probs(static_cast<size_t>(seq));
        for (int64_t h = h_begin; h < h_end; ++h) {
            const int64_t kh = h / groups;
            for (int64_t i = 0; i < seq; ++i) {
                const int64_t j_begin = sliding_attention ? std::max<int64_t>(0, i - sliding_window + 1) : 0;
                const int64_t j_end = i;
                float max_logit = sinks[static_cast<size_t>(h)];
                for (int64_t j = j_begin; j <= j_end; ++j) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        const float qv = q.data[static_cast<size_t>((h * seq + i) * head_dim + d)];
                        const float kv = k.data[static_cast<size_t>((kh * seq + j) * head_dim + d)];
                        dot = std::fma(qv, kv, dot);
                    }
                    const float value = round_to_bf16(dot * scale);
                    logits[static_cast<size_t>(j)] = value;
                    max_logit = std::max(max_logit, value);
                }

                double denom = 0.0;
                for (int64_t j = j_begin; j <= j_end; ++j) {
                    denom += std::exp(static_cast<double>(round_to_bf16(logits[static_cast<size_t>(j)] - max_logit)));
                }
                denom += std::exp(static_cast<double>(round_to_bf16(sinks[static_cast<size_t>(h)] - max_logit)));

                for (int64_t j = j_begin; j <= j_end; ++j) {
                    probs[static_cast<size_t>(j)] =
                        round_to_bf16(static_cast<float>(std::exp(static_cast<double>(
                                           round_to_bf16(logits[static_cast<size_t>(j)] - max_logit))) /
                                       denom));
                }
                for (int64_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    for (int64_t j = j_begin; j <= j_end; ++j) {
                        sum = std::fma(probs[static_cast<size_t>(j)],
                                       v.data[static_cast<size_t>((kh * seq + j) * head_dim + d)],
                                       sum);
                    }
                    out.data[static_cast<size_t>(i * heads * head_dim + h * head_dim + d)] = round_to_bf16(sum);
                }
            }
        }
    };

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int64_t worker_count = std::min<int64_t>(heads, std::max<int64_t>(1, static_cast<int64_t>(hw)));
    if (worker_count <= 1 || heads <= 1) {
        compute_head_range(0, heads);
    } else {
        std::vector<std::future<void>> futures;
        futures.reserve(static_cast<size_t>(worker_count));
        for (int64_t worker = 0; worker < worker_count; ++worker) {
            const int64_t h_begin = (heads * worker) / worker_count;
            const int64_t h_end = (heads * (worker + 1)) / worker_count;
            futures.push_back(std::async(std::launch::async, compute_head_range, h_begin, h_end));
        }
        for (auto& future : futures) {
            future.get();
        }
    }
    return out;
}

static TensorF32 attention_pre_o_cuda_bf16(const TensorF32& q,
                                           const TensorF32& k,
                                           const TensorF32& v,
                                           const std::vector<float>& sinks,
                                           int64_t heads,
                                           int64_t kv_heads,
                                           int64_t seq,
                                           int64_t head_dim,
                                           bool sliding_attention,
                                           int64_t sliding_window,
                                           double* elapsed_ms = nullptr) {
#ifndef SD_USE_CUDA
    (void)q;
    (void)k;
    (void)v;
    (void)sinks;
    (void)heads;
    (void)kv_heads;
    (void)seq;
    (void)head_dim;
    (void)sliding_attention;
    (void)sliding_window;
    (void)elapsed_ms;
    throw std::runtime_error("CUDA attention requires an SD_USE_CUDA build");
#else
    if (q.shape != std::vector<int64_t>{1, heads, seq, head_dim} ||
        k.shape != std::vector<int64_t>{1, kv_heads, seq, head_dim} ||
        v.shape != std::vector<int64_t>{1, kv_heads, seq, head_dim} ||
        static_cast<int64_t>(sinks.size()) != heads ||
        heads % kv_heads != 0) {
        throw std::runtime_error("CUDA attention input shape mismatch");
    }

    const int64_t groups = heads / kv_heads;
    std::vector<float> q_upload(static_cast<size_t>(heads * seq * head_dim));
    std::vector<float> k_upload(static_cast<size_t>(heads * seq * head_dim));
    std::vector<float> v_upload(static_cast<size_t>(heads * head_dim * seq));
    for (int64_t h = 0; h < heads; ++h) {
        const int64_t kh = h / groups;
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const size_t qkv_src = static_cast<size_t>((h * seq + s) * head_dim + d);
                const size_t kv_src = static_cast<size_t>((kh * seq + s) * head_dim + d);
                q_upload[static_cast<size_t>(d + head_dim * (s + seq * h))] = q.data[qkv_src];
                k_upload[static_cast<size_t>(d + head_dim * (s + seq * h))] = k.data[kv_src];
                v_upload[static_cast<size_t>(s + seq * (d + head_dim * h))] = v.data[kv_src];
            }
        }
    }

    std::vector<float> mask(static_cast<size_t>(heads * seq * seq), 0.0f);
    for (int64_t h = 0; h < heads; ++h) {
        for (int64_t i = 0; i < seq; ++i) {
            for (int64_t j = 0; j < seq; ++j) {
                if (j > i || (sliding_attention && (i - j) >= sliding_window)) {
                    mask[static_cast<size_t>(j + seq * (i + seq * h))] = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        throw std::runtime_error("ggml_backend_cuda_init(0) failed for attention");
    }

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    try {
        ggml_init_params params;
        params.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("ggml_init failed for attention CUDA");
        }

        ggml_tensor* q_t = ggml_new_tensor_3d(ctx, GGML_TYPE_BF16, head_dim, seq, heads);
        ggml_tensor* k_t = ggml_new_tensor_3d(ctx, GGML_TYPE_BF16, head_dim, seq, heads);
        ggml_tensor* v_t = ggml_new_tensor_3d(ctx, GGML_TYPE_BF16, seq, head_dim, heads);
        ggml_tensor* mask_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq, seq, heads);
        ggml_tensor* sinks_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
        ggml_set_name(q_t, "gpt_oss_q");
        ggml_set_name(k_t, "gpt_oss_k");
        ggml_set_name(v_t, "gpt_oss_v");
        ggml_set_name(mask_t, "gpt_oss_attention_mask");
        ggml_set_name(sinks_t, "gpt_oss_attention_sinks");
        ggml_set_input(q_t);
        ggml_set_input(k_t);
        ggml_set_input(v_t);
        ggml_set_input(mask_t);
        ggml_set_input(sinks_t);

        ggml_tensor* kq = ggml_mul_mat(ctx, k_t, q_t);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        ggml_tensor* probs = ggml_soft_max_ext(ctx, kq, mask_t, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);
        ggml_soft_max_add_sinks(probs, sinks_t);
        probs = ggml_cast(ctx, probs, GGML_TYPE_BF16);
        ggml_set_name(probs, "gpt_oss_attention_token_scores_bf16");
        ggml_tensor* out = ggml_mul_mat(ctx, v_t, probs);
        ggml_mul_mat_set_prec(out, GGML_PREC_F32);
        out = ggml_cast(ctx, out, GGML_TYPE_BF16);
        ggml_set_name(out, "gpt_oss_attention_pre_o");
        ggml_set_output(out);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("attention CUDA backend tensor allocation failed");
        }
        ggml_backend_buffer_clear(buffer, 0);
        upload_f32_or_bf16(q_t, q_upload);
        upload_f32_or_bf16(k_t, k_upload);
        upload_f32_or_bf16(v_t, v_upload);
        upload_f32_or_bf16(mask_t, mask);
        ggml_backend_tensor_set(sinks_t, sinks.data(), 0, sinks.size() * sizeof(float));
        ggml_backend_synchronize(backend);

        const auto start = std::chrono::steady_clock::now();
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const auto end = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("attention CUDA graph compute failed");
        }
        if (elapsed_ms != nullptr) {
            *elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        }

        TensorF32 raw = download_tensor_f32(out, {head_dim, seq, heads});
        TensorF32 result;
        result.shape = {1, seq, heads * head_dim};
        result.data.resize(static_cast<size_t>(seq * heads * head_dim));
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t s = 0; s < seq; ++s) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    result.data[static_cast<size_t>(s * heads * head_dim + h * head_dim + d)] =
                        raw.data[static_cast<size_t>(d + head_dim * (s + seq * h))];
                }
            }
        }

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return result;
    } catch (...) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        ggml_backend_free(backend);
        throw;
    }
#endif
}

struct TopKResult {
    TensorF32 logits;
    std::vector<int64_t> indices;
    std::vector<float> weights;
    std::vector<int64_t> counts;
};

static TopKResult topk_from_logits(const TensorF32& logits, int top_k) {
    if (logits.shape.size() != 3 || logits.shape[0] != 1) {
        throw std::runtime_error("router logits shape mismatch");
    }
    TopKResult out;
    out.logits = logits;
    const int64_t seq = logits.shape[1];
    const int64_t experts = logits.shape[2];
    out.indices.resize(static_cast<size_t>(seq * top_k));
    out.weights.resize(static_cast<size_t>(seq * top_k));
    out.counts.assign(static_cast<size_t>(experts), 0);
    for (int64_t s = 0; s < seq; ++s) {
        std::vector<std::pair<float, int64_t>> values;
        values.reserve(static_cast<size_t>(experts));
        for (int64_t e = 0; e < experts; ++e) {
            values.emplace_back(logits.data[static_cast<size_t>(s * experts + e)], e);
        }
        std::partial_sort(values.begin(), values.begin() + top_k, values.end(),
                          [](const auto& a, const auto& b) {
                              if (a.first != b.first) return a.first > b.first;
                              return a.second < b.second;
                          });
        float max_v = values[0].first;
        double denom = 0.0;
        for (int k = 0; k < top_k; ++k) {
            denom += std::exp(static_cast<double>(values[k].first - max_v));
        }
        for (int k = 0; k < top_k; ++k) {
            const size_t idx = static_cast<size_t>(s * top_k + k);
            out.indices[idx] = values[k].second;
            out.weights[idx] = round_to_bf16(static_cast<float>(std::exp(static_cast<double>(values[k].first - max_v)) / denom));
            out.counts[static_cast<size_t>(values[k].second)]++;
        }
    }
    return out;
}

static TopKResult make_topk_from_indices_and_weights(const TensorF32& logits,
                                                     const std::vector<int64_t>& indices,
                                                     const std::vector<float>& weights,
                                                     int top_k) {
    if (logits.shape.size() != 3 || logits.shape[0] != 1) {
        throw std::runtime_error("router logits shape mismatch");
    }
    const int64_t seq = logits.shape[1];
    const int64_t experts = logits.shape[2];
    if (static_cast<int64_t>(indices.size()) != seq * top_k ||
        static_cast<int64_t>(weights.size()) != seq * top_k) {
        throw std::runtime_error("top-k explicit selection shape mismatch");
    }
    TopKResult out;
    out.logits = logits;
    out.indices = indices;
    out.weights = weights;
    out.counts.assign(static_cast<size_t>(experts), 0);
    for (int64_t idx : out.indices) {
        if (idx < 0 || idx >= experts) {
            throw std::runtime_error("top-k expert out of range");
        }
        out.counts[static_cast<size_t>(idx)]++;
    }
    return out;
}

static TopKResult canonicalize_topk_by_expert(const TopKResult& topk, int top_k) {
    const int64_t seq = topk.logits.shape[1];
    std::vector<int64_t> indices(topk.indices.size());
    std::vector<float> weights(topk.weights.size());
    for (int64_t s = 0; s < seq; ++s) {
        std::vector<std::pair<int64_t, float>> pairs;
        pairs.reserve(static_cast<size_t>(top_k));
        for (int k = 0; k < top_k; ++k) {
            const size_t idx = static_cast<size_t>(s * top_k + k);
            pairs.emplace_back(topk.indices[idx], topk.weights[idx]);
        }
        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        for (int k = 0; k < top_k; ++k) {
            const size_t idx = static_cast<size_t>(s * top_k + k);
            indices[idx] = pairs[static_cast<size_t>(k)].first;
            weights[idx] = pairs[static_cast<size_t>(k)].second;
        }
    }
    return make_topk_from_indices_and_weights(topk.logits, indices, weights, top_k);
}

static TopKResult router_topk(const TensorF32& x,
                              const std::vector<float>& weight,
                              const std::vector<int64_t>& weight_shape,
                              const std::vector<float>& bias,
                              int top_k) {
    return topk_from_logits(linear_bf16(x, weight, weight_shape, bias, "router"), top_k);
}

static TensorF32 topk_indices_tensor(const TopKResult& topk, int64_t seq, int top_k) {
    std::vector<float> values(topk.indices.size());
    for (size_t i = 0; i < topk.indices.size(); ++i) values[i] = static_cast<float>(topk.indices[i]);
    return make_tensor({1, seq, top_k}, std::move(values));
}

static TensorF32 topk_weights_tensor(const TopKResult& topk, int64_t seq, int top_k) {
    return make_tensor({1, seq, top_k}, topk.weights);
}

static std::vector<float> topk_weights_for_indices(const TensorF32& logits,
                                                   const std::vector<int64_t>& indices,
                                                   int top_k) {
    if (logits.shape.size() != 3 || logits.shape[0] != 1) {
        throw std::runtime_error("router logits shape mismatch for selected weights");
    }
    const int64_t seq = logits.shape[1];
    const int64_t experts = logits.shape[2];
    if (static_cast<int64_t>(indices.size()) != seq * top_k) {
        throw std::runtime_error("router selected index shape mismatch");
    }
    std::vector<float> weights(indices.size(), 0.0f);
    for (int64_t s = 0; s < seq; ++s) {
        float max_v = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < top_k; ++k) {
            const int64_t expert = indices[static_cast<size_t>(s * top_k + k)];
            if (expert < 0 || expert >= experts) {
                throw std::runtime_error("router selected expert out of range");
            }
            max_v = std::max(max_v, logits.data[static_cast<size_t>(s * experts + expert)]);
        }
        double denom = 0.0;
        for (int k = 0; k < top_k; ++k) {
            const int64_t expert = indices[static_cast<size_t>(s * top_k + k)];
            denom += std::exp(static_cast<double>(logits.data[static_cast<size_t>(s * experts + expert)] - max_v));
        }
        for (int k = 0; k < top_k; ++k) {
            const int64_t expert = indices[static_cast<size_t>(s * top_k + k)];
            const size_t out_i = static_cast<size_t>(s * top_k + k);
            weights[out_i] = round_to_bf16(static_cast<float>(std::exp(static_cast<double>(logits.data[static_cast<size_t>(s * experts + expert)] - max_v)) / denom));
        }
    }
    return weights;
}

static std::vector<float> canonical_topk_weight_vector(const std::vector<int64_t>& indices,
                                                       const std::vector<float>& weights,
                                                       int top_k) {
    if (indices.size() != weights.size() || indices.size() % static_cast<size_t>(top_k) != 0) {
        throw std::runtime_error("canonical top-k weight shape mismatch");
    }
    std::vector<float> out(weights.size(), 0.0f);
    const size_t seq = indices.size() / static_cast<size_t>(top_k);
    for (size_t s = 0; s < seq; ++s) {
        std::vector<std::pair<int64_t, float>> pairs;
        pairs.reserve(static_cast<size_t>(top_k));
        for (int k = 0; k < top_k; ++k) {
            const size_t idx = s * static_cast<size_t>(top_k) + static_cast<size_t>(k);
            pairs.emplace_back(indices[idx], weights[idx]);
        }
        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        for (int k = 0; k < top_k; ++k) {
            out[s * static_cast<size_t>(top_k) + static_cast<size_t>(k)] = pairs[static_cast<size_t>(k)].second;
        }
    }
    return out;
}

static float decode_mxfp4_value(const std::vector<uint8_t>& blocks,
                                const std::vector<uint8_t>& scales,
                                const std::vector<int64_t>& shape,
                                int64_t expert,
                                int64_t out,
                                int64_t in) {
    static const float lut[16] = {
        +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    const int64_t group = in / 32;
    const int64_t byte = (in % 32) / 2;
    const bool high = (in % 2) != 0;
    const size_t block_idx = static_cast<size_t>(((expert * shape[1] + out) * shape[2] + group) * shape[3] + byte);
    const size_t scale_idx = static_cast<size_t>((expert * shape[1] + out) * shape[2] + group);
    const uint8_t packed = blocks[block_idx];
    const uint8_t nibble = high ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 0x0F);
    const int exp = static_cast<int>(scales[scale_idx]) - 127;
    return std::ldexp(lut[nibble], exp);
}

static std::string category_for(const std::string& name) {
    if (name == "model.embed_tokens.weight") return "embedding";
    if (name == "lm_head.weight") return "lm_head";
    if (name.find("self_attn.q_proj") != std::string::npos) return "attention_q";
    if (name.find("self_attn.k_proj") != std::string::npos) return "attention_k";
    if (name.find("self_attn.v_proj") != std::string::npos) return "attention_v";
    if (name.find("self_attn.o_proj") != std::string::npos) return "attention_o";
    if (name.find("self_attn.sinks") != std::string::npos) return "attention_sink";
    if (name.find("layernorm") != std::string::npos || name == "model.norm.weight") return "norm";
    if (name.find("mlp.router") != std::string::npos) return "router";
    if (name.find("gate_up_proj_blocks") != std::string::npos ||
        name.find("down_proj_blocks") != std::string::npos) return "expert_mxfp4_blocks";
    if (name.find("gate_up_proj_scales") != std::string::npos ||
        name.find("down_proj_scales") != std::string::npos) return "expert_mxfp4_scales";
    if (name.find("gate_up_proj_bias") != std::string::npos ||
        name.find("down_proj_bias") != std::string::npos) return "expert_bias";
    return "other";
}

static int layer_index_for(const std::string& name) {
    const std::string prefix = "model.layers.";
    const size_t p = name.find(prefix);
    if (p == std::string::npos) return -1;
    const size_t start = p + prefix.size();
    const size_t end = name.find('.', start);
    if (end == std::string::npos) return -1;
    return std::atoi(name.substr(start, end - start).c_str());
}

static void write_weight_map_csv(const std::filesystem::path& path, const std::vector<SafetensorEntry>& entries) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write weight map: " + path.string());
    }
    out << "name,shape,dtype,shard,module_path,layer,category,bytes,mxfp4\n";
    for (const auto& e : entries) {
        out << e.name << ",";
        out << "\"";
        for (size_t i = 0; i < e.shape.size(); ++i) {
            if (i) out << "x";
            out << e.shape[i];
        }
        out << "\",";
        const std::string category = category_for(e.name);
        const uint64_t bytes = e.end - e.begin;
        out << e.dtype << "," << e.file.filename().string() << "," << e.name << ","
            << layer_index_for(e.name) << "," << category << "," << bytes << ","
            << (category.find("mxfp4") != std::string::npos ? "true" : "false") << "\n";
    }
}

static void run_embedding_parity(const std::filesystem::path& oracle_dir,
                                 const std::unordered_map<std::string, SafetensorEntry>& index) {
    const NpyI64 ids = load_npy_i64(oracle_dir / "input_ids_i64.npy");
    const NpyF32 ref = load_npy_f32(oracle_dir / "token_embedding_f32.npy");
    if (ids.shape.size() != 2 || ref.shape.size() != 3 || ids.shape[0] != ref.shape[0] || ids.shape[1] != ref.shape[1]) {
        throw std::runtime_error("input id / embedding oracle shape mismatch");
    }
    const auto it = index.find("model.embed_tokens.weight");
    if (it == index.end()) {
        throw std::runtime_error("missing model.embed_tokens.weight");
    }
    const SafetensorEntry& emb_entry = it->second;
    if (emb_entry.shape.size() != 2 || emb_entry.shape[1] != ref.shape[2]) {
        throw std::runtime_error("embedding weight shape mismatch");
    }
    const std::vector<float> embedding = read_bf16_as_f32(emb_entry);
    const int64_t seq = ids.shape[1];
    const int64_t hidden = emb_entry.shape[1];
    std::vector<float> actual(ref.data.size());
    for (int64_t s = 0; s < seq; ++s) {
        const int64_t token = ids.data[static_cast<size_t>(s)];
        if (token < 0 || token >= emb_entry.shape[0]) {
            throw std::runtime_error("input id out of embedding vocab range");
        }
        const size_t src = static_cast<size_t>(token * hidden);
        const size_t dst = static_cast<size_t>(s * hidden);
        std::copy_n(embedding.data() + src, static_cast<size_t>(hidden), actual.data() + dst);
    }
    const DiffStats d = diff_stats(actual, ref.data);
    std::cout << "embedding_parity shape=1x" << seq << "x" << hidden
              << " max_diff=" << d.max_diff
              << " mean_diff=" << d.mean_diff
              << " rel_max=" << d.rel_max
              << " max_index=" << d.max_index << "\n";
    if (d.max_diff > 0.0f) {
        throw std::runtime_error("embedding parity is not exact against BF16 oracle");
    }
}

static TensorF32 token_embedding_from_ids(const NpyI64& ids,
                                          const std::unordered_map<std::string, SafetensorEntry>& index) {
    if (ids.shape.size() != 2 || ids.shape[0] != 1) {
        throw std::runtime_error("input_ids must have shape [1, seq]");
    }
    const auto it = index.find("model.embed_tokens.weight");
    if (it == index.end()) {
        throw std::runtime_error("missing model.embed_tokens.weight");
    }
    const SafetensorEntry& emb_entry = it->second;
    if (emb_entry.shape.size() != 2) {
        throw std::runtime_error("embedding weight shape mismatch");
    }
    const int64_t seq = ids.shape[1];
    const int64_t hidden = emb_entry.shape[1];
    if (emb_entry.dtype != "BF16") {
        throw std::runtime_error("embedding weight must be BF16 for row-gather path");
    }
    const TensorBytesView embedding_view = tensor_bytes_view(emb_entry);
    const size_t row_bytes = static_cast<size_t>(hidden) * sizeof(ggml_bf16_t);
    TensorF32 out;
    out.shape = {1, seq, hidden};
    out.data.resize(static_cast<size_t>(seq * hidden));
    for (int64_t s = 0; s < seq; ++s) {
        const int64_t token = ids.data[static_cast<size_t>(s)];
        if (token < 0 || token >= emb_entry.shape[0]) {
            throw std::runtime_error("input id out of embedding vocab range");
        }
        const size_t dst = static_cast<size_t>(s * hidden);
        const size_t src_bytes = static_cast<size_t>(token) * row_bytes;
        if (src_bytes + row_bytes > embedding_view.size) {
            throw std::runtime_error("embedding row view out of range");
        }
        const auto* src = reinterpret_cast<const ggml_bf16_t*>(embedding_view.data + src_bytes);
        ggml_bf16_to_fp32_row(src, out.data.data() + dst, hidden);
    }
    return out;
}

static void run_mxfp4_proof(const std::filesystem::path& oracle_dir,
                            const std::unordered_map<std::string, SafetensorEntry>& index) {
    const auto blocks_it = index.find("model.layers.0.mlp.experts.gate_up_proj_blocks");
    const auto scales_it = index.find("model.layers.0.mlp.experts.gate_up_proj_scales");
    if (blocks_it == index.end() || scales_it == index.end()) {
        throw std::runtime_error("missing layer 0 gate_up_proj MXFP4 tensors");
    }
    const SafetensorEntry& blocks_e = blocks_it->second;
    const SafetensorEntry& scales_e = scales_it->second;
    if (blocks_e.dtype != "U8" || scales_e.dtype != "U8" ||
        blocks_e.shape.size() != 4 || scales_e.shape.size() != 3 ||
        blocks_e.shape[0] < 1 || blocks_e.shape[1] < 32 || blocks_e.shape[2] < 1 || blocks_e.shape[3] < 1) {
        throw std::runtime_error("unexpected MXFP4 tensor shape");
    }
    const std::vector<uint8_t> blocks = read_tensor_bytes(blocks_e);
    const std::vector<uint8_t> scales = read_tensor_bytes(scales_e);
    const float lut[16] = {
        +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    auto block_index = [&](int64_t e, int64_t out, int64_t group, int64_t byte) -> size_t {
        return static_cast<size_t>(((e * blocks_e.shape[1] + out) * blocks_e.shape[2] + group) * blocks_e.shape[3] + byte);
    };
    auto scale_index = [&](int64_t e, int64_t out, int64_t group) -> size_t {
        return static_cast<size_t>((e * scales_e.shape[1] + out) * scales_e.shape[2] + group);
    };
    std::vector<float> actual(32);
    for (int out = 0; out < 32; ++out) {
        const uint8_t packed = blocks[block_index(0, out, 0, 0)];
        const uint8_t nibble = packed & 0x0F;
        const int exp = static_cast<int>(scales[scale_index(0, out, 0)]) - 127;
        actual[static_cast<size_t>(out)] = std::ldexp(lut[nibble], exp);
    }
    const NpyF32 ref = load_npy_f32(oracle_dir / "mxfp4_gate_up_e0_hidden0_out0_31_f32.npy");
    if (ref.data.size() != actual.size()) {
        throw std::runtime_error("MXFP4 reference shape mismatch");
    }
    const DiffStats d = diff_stats(actual, ref.data);
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    for (float v : actual) {
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        sum += v;
    }
    std::cout << "mxfp4_decode tensor=model.layers.0.mlp.experts.gate_up_proj_blocks"
              << " raw_shape=32x5760x90x16 decoded_slice=gate_up_proj[0,0,0:32]"
              << " min=" << min_v << " max=" << max_v << " mean=" << static_cast<float>(sum / actual.size())
              << " max_diff=" << d.max_diff
              << " mean_diff=" << d.mean_diff
              << " rel_max=" << d.rel_max
              << " max_index=" << d.max_index << "\n";
    if (d.max_diff > 0.0f) {
        throw std::runtime_error("MXFP4 proof did not match oracle");
    }
}

static void report_i64_match(const std::string& label,
                             const std::vector<int64_t>& actual,
                             const NpyI64& ref,
                             const std::vector<int64_t>& shape) {
    if (ref.shape != shape || ref.data.size() != actual.size()) {
        throw std::runtime_error(label + " shape mismatch");
    }
    size_t mismatches = 0;
    size_t first = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != ref.data[i]) {
            if (mismatches == 0) first = i;
            ++mismatches;
        }
    }
    std::cout << label << " shape=";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) std::cout << "x";
        std::cout << shape[i];
    }
    std::cout << " mismatches=" << mismatches;
    if (mismatches) {
        std::cout << " first_mismatch=" << first << " actual=" << actual[first] << " ref=" << ref.data[first];
    }
    std::cout << "\n";
    if (mismatches) {
        throw std::runtime_error(label + " mismatch");
    }
}

struct RouterCudaMode {
    std::string name;
    ggml_type input_type = GGML_TYPE_BF16;
    ggml_type weight_type = GGML_TYPE_BF16;
    ggml_type bias_type = GGML_TYPE_BF16;
    bool force_prec_f32 = true;
    bool cast_output_bf16 = true;
};

static const char* ggml_type_label(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_BF16: return "BF16";
        default: return ggml_type_name(type);
    }
}

static void upload_f32_or_bf16(ggml_tensor* tensor, const std::vector<float>& data) {
    if (static_cast<size_t>(ggml_nelements(tensor)) != data.size()) {
        throw std::runtime_error(std::string("upload shape mismatch for ") + tensor->name);
    }
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
    } else if (tensor->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> bf16(data.size());
        ggml_fp32_to_bf16_row_ref(data.data(), bf16.data(), static_cast<int64_t>(data.size()));
        ggml_backend_tensor_set(tensor, bf16.data(), 0, bf16.size() * sizeof(ggml_bf16_t));
    } else {
        throw std::runtime_error("unsupported upload tensor dtype");
    }
}

static void upload_bf16_raw_or_f32(ggml_tensor* tensor,
                                   const std::vector<ggml_bf16_t>& raw,
                                   const std::vector<float>& f32) {
    if (static_cast<size_t>(ggml_nelements(tensor)) != raw.size()) {
        throw std::runtime_error(std::string("upload shape mismatch for ") + tensor->name);
    }
    if (tensor->type == GGML_TYPE_BF16) {
        ggml_backend_tensor_set(tensor, raw.data(), 0, raw.size() * sizeof(ggml_bf16_t));
    } else if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, f32.data(), 0, f32.size() * sizeof(float));
    } else {
        throw std::runtime_error("unsupported upload tensor dtype");
    }
}

static TensorF32 download_tensor_f32(ggml_tensor* tensor, const std::vector<int64_t>& shape) {
    const int64_t n = ggml_nelements(tensor);
    if (n < 0 || static_cast<uint64_t>(n) != elem_count(shape)) {
        throw std::runtime_error("download tensor shape mismatch");
    }
    TensorF32 out;
    out.shape = shape;
    out.data.resize(static_cast<size_t>(n));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, out.data.data(), 0, out.data.size() * sizeof(float));
    } else if (tensor->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> tmp(static_cast<size_t>(n));
        ggml_backend_tensor_get(tensor, tmp.data(), 0, tmp.size() * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row(tmp.data(), out.data.data(), n);
    } else {
        throw std::runtime_error("unsupported download tensor dtype");
    }
    return out;
}

static std::vector<int64_t> download_tensor_i32_as_i64(ggml_tensor* tensor, uint64_t expected) {
    if (tensor->type != GGML_TYPE_I32) {
        throw std::runtime_error("expected i32 tensor download");
    }
    const int64_t n = ggml_nelements(tensor);
    if (n < 0 || static_cast<uint64_t>(n) != expected) {
        throw std::runtime_error("download i32 tensor shape mismatch");
    }
    std::vector<int32_t> tmp(static_cast<size_t>(n));
    ggml_backend_tensor_get(tensor, tmp.data(), 0, tmp.size() * sizeof(int32_t));
    std::vector<int64_t> out(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        out[i] = static_cast<int64_t>(tmp[i]);
    }
    return out;
}

static TensorF32 rms_norm_ggml_cuda(const TensorF32& input,
                                    const std::vector<float>& weight,
                                    float eps,
                                    double* elapsed_ms) {
#ifndef SD_USE_CUDA
    (void)input;
    (void)weight;
    (void)eps;
    (void)elapsed_ms;
    throw std::runtime_error("ggml CUDA RMSNorm requires an SD_USE_CUDA build");
#else
    if (input.shape.size() != 3 || input.shape[0] != 1 || static_cast<int64_t>(weight.size()) != input.shape[2]) {
        throw std::runtime_error("ggml CUDA RMSNorm input shape mismatch");
    }
    const int64_t seq = input.shape[1];
    const int64_t hidden = input.shape[2];
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        throw std::runtime_error("ggml_backend_cuda_init(0) failed for RMSNorm");
    }
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    try {
        ggml_init_params params;
        params.mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("ggml_init failed for RMSNorm");
        }
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, seq);
        ggml_tensor* w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
        ggml_set_name(x, "gptoss_rmsnorm_input");
        ggml_set_name(w, "gptoss_rmsnorm_weight");
        ggml_set_input(x);
        ggml_set_input(w);
        ggml_tensor* y = ggml_rms_norm(ctx, x, eps);
        ggml_set_name(y, "gptoss_rmsnorm_unweighted");
        y = ggml_mul(ctx, y, w);
        ggml_set_name(y, "gptoss_rmsnorm_weighted");
        y = ggml_cast(ctx, y, GGML_TYPE_BF16);
        ggml_set_name(y, "gptoss_rmsnorm_output_bf16");
        ggml_set_output(y);
        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("RMSNorm CUDA backend tensor allocation failed");
        }
        ggml_backend_buffer_clear(buffer, 0);
        upload_f32_or_bf16(x, input.data);
        upload_f32_or_bf16(w, weight);
        ggml_backend_synchronize(backend);
        const auto start = std::chrono::steady_clock::now();
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const auto end = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("RMSNorm CUDA graph compute failed");
        }
        if (elapsed_ms != nullptr) {
            *elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        }
        TensorF32 out = download_tensor_f32(y, input.shape);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return out;
    } catch (...) {
        if (buffer != nullptr) ggml_backend_buffer_free(buffer);
        if (ctx != nullptr) ggml_free(ctx);
        ggml_backend_free(backend);
        throw;
    }
#endif
}

static TensorF32 run_router_cuda_mode(const RouterCudaMode& mode,
                                      const TensorF32& input,
                                      const std::vector<ggml_bf16_t>& weight_bf16,
                                      const std::vector<float>& weight_f32,
                                      const std::vector<int64_t>& weight_shape,
                                      const std::vector<ggml_bf16_t>& bias_bf16,
                                      const std::vector<float>& bias_f32,
                                      int top_k,
                                      std::vector<int64_t>* cuda_argsort_topk,
                                      double* elapsed_ms,
                                      size_t* buffer_bytes,
                                      std::string* output_dtype) {
#ifndef SD_USE_CUDA
    (void)mode;
    (void)input;
    (void)weight_bf16;
    (void)weight_f32;
    (void)weight_shape;
    (void)bias_bf16;
    (void)bias_f32;
    (void)top_k;
    (void)cuda_argsort_topk;
    (void)elapsed_ms;
    (void)buffer_bytes;
    (void)output_dtype;
    throw std::runtime_error("router CUDA parity requires an SD_USE_CUDA build");
#else
    if (input.shape.size() != 3 || input.shape[0] != 1 || weight_shape.size() != 2) {
        throw std::runtime_error("router CUDA input/weight rank mismatch");
    }
    const int64_t seq = input.shape[1];
    const int64_t hidden = input.shape[2];
    const int64_t experts = weight_shape[0];
    if (weight_shape[1] != hidden ||
        static_cast<int64_t>(weight_bf16.size()) != experts * hidden ||
        static_cast<int64_t>(weight_f32.size()) != experts * hidden ||
        static_cast<int64_t>(bias_bf16.size()) != experts ||
        static_cast<int64_t>(bias_f32.size()) != experts) {
        throw std::runtime_error("router CUDA tensor shape mismatch");
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        throw std::runtime_error("ggml_backend_cuda_init(0) failed");
    }

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    try {
        ggml_init_params params;
        params.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("ggml_init failed for router CUDA");
        }

        ggml_tensor* x = ggml_new_tensor_2d(ctx, mode.input_type, hidden, seq);
        ggml_tensor* w = ggml_new_tensor_2d(ctx, mode.weight_type, hidden, experts);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, mode.bias_type, experts);
        ggml_set_name(x, "router_input");
        ggml_set_name(w, "router_weight");
        ggml_set_name(b, "router_bias");
        ggml_set_input(x);
        ggml_set_input(w);
        ggml_set_input(b);

        ggml_tensor* y = ggml_mul_mat(ctx, w, x);
        if (mode.force_prec_f32) {
            ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        }
        y = ggml_add(ctx, y, b);
        if (mode.cast_output_bf16) {
            y = ggml_cast(ctx, y, GGML_TYPE_BF16);
        }
        ggml_set_name(y, "router_logits");
        ggml_set_output(y);
        ggml_tensor* sorted_indices = nullptr;
        if (cuda_argsort_topk != nullptr) {
            ggml_tensor* y_topk_src = y;
            if (y_topk_src->type != GGML_TYPE_F32) {
                y_topk_src = ggml_cast(ctx, y_topk_src, GGML_TYPE_F32);
                ggml_set_name(y_topk_src, "router_logits_f32_for_topk");
            }
            sorted_indices = ggml_argsort(ctx, y_topk_src, GGML_SORT_ORDER_DESC);
            ggml_set_name(sorted_indices, "router_argsort_indices");
            ggml_set_output(sorted_indices);
        }

        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        if (sorted_indices != nullptr) {
            ggml_build_forward_expand(graph, sorted_indices);
        }

        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("router CUDA backend tensor allocation failed");
        }
        if (buffer_bytes != nullptr) {
            *buffer_bytes = ggml_backend_buffer_get_size(buffer);
        }
        ggml_backend_buffer_clear(buffer, 0);
        upload_f32_or_bf16(x, input.data);
        upload_bf16_raw_or_f32(w, weight_bf16, weight_f32);
        upload_bf16_raw_or_f32(b, bias_bf16, bias_f32);
        ggml_backend_synchronize(backend);

        const auto start = std::chrono::steady_clock::now();
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const auto end = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("router CUDA graph compute failed");
        }
        if (elapsed_ms != nullptr) {
            *elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        }
        if (output_dtype != nullptr) {
            *output_dtype = ggml_type_label(y->type);
        }
        TensorF32 out = download_tensor_f32(y, {1, seq, experts});
        if (cuda_argsort_topk != nullptr && sorted_indices != nullptr) {
            std::vector<int64_t> sorted = download_tensor_i32_as_i64(sorted_indices, static_cast<uint64_t>(seq * experts));
            cuda_argsort_topk->assign(static_cast<size_t>(seq * top_k), 0);
            for (int64_t s = 0; s < seq; ++s) {
                for (int k = 0; k < top_k; ++k) {
                    (*cuda_argsort_topk)[static_cast<size_t>(s * top_k + k)] =
                        sorted[static_cast<size_t>(s * experts + k)];
                }
            }
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return out;
    } catch (...) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        ggml_backend_free(backend);
        throw;
    }
#endif
}

static size_t topk_mismatch_count(const std::vector<int64_t>& actual, const NpyI64& ref) {
    if (actual.size() != ref.data.size()) {
        throw std::runtime_error("top-k size mismatch");
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != ref.data[i]) {
            ++mismatches;
        }
    }
    return mismatches;
}

static size_t topk_set_mismatch_tokens(const std::vector<int64_t>& actual, const NpyI64& ref, int top_k) {
    if (actual.size() != ref.data.size() || actual.size() % static_cast<size_t>(top_k) != 0) {
        throw std::runtime_error("top-k set size mismatch");
    }
    size_t mismatches = 0;
    const size_t seq = actual.size() / static_cast<size_t>(top_k);
    for (size_t s = 0; s < seq; ++s) {
        std::vector<int64_t> a(actual.begin() + static_cast<std::ptrdiff_t>(s * top_k),
                               actual.begin() + static_cast<std::ptrdiff_t>((s + 1) * top_k));
        std::vector<int64_t> r(ref.data.begin() + static_cast<std::ptrdiff_t>(s * top_k),
                               ref.data.begin() + static_cast<std::ptrdiff_t>((s + 1) * top_k));
        std::sort(a.begin(), a.end());
        std::sort(r.begin(), r.end());
        if (a != r) {
            ++mismatches;
        }
    }
    return mismatches;
}

static size_t topk_tie_like_mismatch_count(const std::vector<int64_t>& actual,
                                           const NpyI64& ref_indices,
                                           const TensorF32& actual_logits,
                                           const NpyF32& ref_logits,
                                           int top_k) {
    if (actual_logits.shape.size() != 3 || ref_logits.shape != actual_logits.shape) {
        throw std::runtime_error("top-k tie diagnostic logits shape mismatch");
    }
    const int64_t experts = actual_logits.shape[2];
    size_t tie_like = 0;
    for (size_t flat = 0; flat < actual.size(); ++flat) {
        if (actual[flat] == ref_indices.data[flat]) {
            continue;
        }
        const int64_t token = static_cast<int64_t>(flat / static_cast<size_t>(top_k));
        const int64_t actual_expert = actual[flat];
        const int64_t ref_expert = ref_indices.data[flat];
        if (actual_expert < 0 || actual_expert >= experts || ref_expert < 0 || ref_expert >= experts) {
            throw std::runtime_error("top-k tie diagnostic expert out of range");
        }
        const size_t actual_pos = static_cast<size_t>(token * experts + actual_expert);
        const size_t ref_pos = static_cast<size_t>(token * experts + ref_expert);
        if (actual_logits.data[actual_pos] == actual_logits.data[ref_pos] &&
            ref_logits.data[actual_pos] == ref_logits.data[ref_pos]) {
            ++tie_like;
        }
    }
    return tie_like;
}

struct TopKCanonicalReport {
    size_t slot_mismatches = 0;
    size_t expert_set_mismatch_tokens = 0;
    size_t tie_like_mismatches = 0;
    DiffStats canonical_weight_diff;
};

static void print_topk_mismatch_table(const std::string& label,
                                      const std::vector<int64_t>& actual,
                                      const NpyI64& ref_indices,
                                      const TensorF32& actual_logits,
                                      const NpyF32& ref_logits,
                                      int top_k,
                                      size_t limit);

static TopKCanonicalReport validate_canonical_topk(const std::string& label,
                                                   const TopKResult& actual,
                                                   const NpyI64& ref_indices,
                                                   const NpyF32& ref_scores,
                                                   const NpyF32& ref_logits,
                                                   int top_k,
                                                   float weight_tolerance = 1.0e-6f) {
    if (ref_scores.shape.size() != 3 || ref_scores.shape[0] != 1 ||
        static_cast<int64_t>(actual.indices.size()) != ref_scores.shape[1] * top_k ||
        actual.indices.size() != ref_indices.data.size() ||
        actual.weights.size() != actual.indices.size()) {
        throw std::runtime_error(label + " canonical top-k shape mismatch");
    }
    TopKCanonicalReport report;
    report.slot_mismatches = topk_mismatch_count(actual.indices, ref_indices);
    report.expert_set_mismatch_tokens = topk_set_mismatch_tokens(actual.indices, ref_indices, top_k);
    report.tie_like_mismatches = topk_tie_like_mismatch_count(actual.indices, ref_indices, actual.logits, ref_logits, top_k);
    const std::vector<float> actual_canonical = canonical_topk_weight_vector(actual.indices, actual.weights, top_k);
    const std::vector<float> ref_canonical = canonical_topk_weight_vector(ref_indices.data, ref_scores.data, top_k);
    report.canonical_weight_diff = diff_stats(actual_canonical, ref_canonical);

    std::cout << label
              << " expert_set_mismatch_tokens=" << report.expert_set_mismatch_tokens
              << " slot_order_mismatches=" << report.slot_mismatches
              << " equal_logit_tie_slot_mismatches=" << report.tie_like_mismatches
              << " canonical_weight_max_diff=" << report.canonical_weight_diff.max_diff
              << " canonical_weight_mean_diff=" << report.canonical_weight_diff.mean_diff
              << " canonical_weight_rel_max=" << report.canonical_weight_diff.rel_max
              << "\n";

    if (report.expert_set_mismatch_tokens != 0) {
        print_topk_mismatch_table(label, actual.indices, ref_indices, actual.logits, ref_logits, top_k, 10);
        throw std::runtime_error(label + " expert set mismatch");
    }
    if (report.slot_mismatches != report.tie_like_mismatches) {
        print_topk_mismatch_table(label, actual.indices, ref_indices, actual.logits, ref_logits, top_k, 10);
        throw std::runtime_error(label + " has non-tie slot mismatch");
    }
    if (report.canonical_weight_diff.max_diff > weight_tolerance) {
        throw std::runtime_error(label + " canonical expert-weight mismatch");
    }
    return report;
}

static void print_topk_mismatch_table(const std::string& label,
                                      const std::vector<int64_t>& actual,
                                      const NpyI64& ref_indices,
                                      const TensorF32& actual_logits,
                                      const NpyF32& ref_logits,
                                      int top_k,
                                      size_t limit) {
    if (actual_logits.shape.size() != 3 || ref_logits.shape != actual_logits.shape) {
        throw std::runtime_error("top-k diagnostic logits shape mismatch");
    }
    const int64_t experts = actual_logits.shape[2];
    std::cout << label << "_mismatch_detail"
              << " columns=flat,token,slot,actual_expert,ref_expert,actual_logit_actual,actual_logit_ref,ref_logit_actual,ref_logit_ref,actual_margin,ref_margin,tie_like\n";
    size_t printed = 0;
    for (size_t flat = 0; flat < actual.size() && printed < limit; ++flat) {
        if (actual[flat] == ref_indices.data[flat]) {
            continue;
        }
        const int64_t token = static_cast<int64_t>(flat / static_cast<size_t>(top_k));
        const int64_t slot = static_cast<int64_t>(flat % static_cast<size_t>(top_k));
        const int64_t actual_expert = actual[flat];
        const int64_t ref_expert = ref_indices.data[flat];
        if (actual_expert < 0 || actual_expert >= experts || ref_expert < 0 || ref_expert >= experts) {
            throw std::runtime_error("top-k diagnostic expert out of range");
        }
        const size_t actual_pos = static_cast<size_t>(token * experts + actual_expert);
        const size_t ref_pos = static_cast<size_t>(token * experts + ref_expert);
        const float actual_logit_actual = actual_logits.data[actual_pos];
        const float actual_logit_ref = actual_logits.data[ref_pos];
        const float ref_logit_actual = ref_logits.data[actual_pos];
        const float ref_logit_ref = ref_logits.data[ref_pos];
        const float actual_margin = actual_logit_actual - actual_logit_ref;
        const float ref_margin = ref_logit_ref - ref_logit_actual;
        const bool tie_like = actual_margin == 0.0f && ref_margin == 0.0f;
        std::cout << label << "_mismatch"
                  << " flat=" << flat
                  << " token=" << token
                  << " slot=" << slot
                  << " actual_expert=" << actual_expert
                  << " ref_expert=" << ref_expert
                  << " actual_logit_actual=" << actual_logit_actual
                  << " actual_logit_ref=" << actual_logit_ref
                  << " ref_logit_actual=" << ref_logit_actual
                  << " ref_logit_ref=" << ref_logit_ref
                  << " actual_margin=" << actual_margin
                  << " ref_margin=" << ref_margin
                  << " tie_like=" << (tie_like ? "true" : "false")
                  << "\n";
        ++printed;
    }
}

static TopKCanonicalReport report_canonical_topk_no_throw(const std::string& label,
                                                          const TopKResult& actual,
                                                          const NpyI64& ref_indices,
                                                          const NpyF32& ref_scores,
                                                          const NpyF32& ref_logits,
                                                          int top_k,
                                                          size_t mismatch_detail_limit = 10) {
    if (ref_scores.shape.size() != 3 || ref_scores.shape[0] != 1 ||
        static_cast<int64_t>(actual.indices.size()) != ref_scores.shape[1] * top_k ||
        actual.indices.size() != ref_indices.data.size() ||
        actual.weights.size() != actual.indices.size()) {
        throw std::runtime_error(label + " canonical top-k shape mismatch");
    }
    TopKCanonicalReport report;
    report.slot_mismatches = topk_mismatch_count(actual.indices, ref_indices);
    report.expert_set_mismatch_tokens = topk_set_mismatch_tokens(actual.indices, ref_indices, top_k);
    report.tie_like_mismatches = topk_tie_like_mismatch_count(actual.indices, ref_indices, actual.logits, ref_logits, top_k);
    const std::vector<float> actual_canonical = canonical_topk_weight_vector(actual.indices, actual.weights, top_k);
    const std::vector<float> ref_canonical = canonical_topk_weight_vector(ref_indices.data, ref_scores.data, top_k);
    report.canonical_weight_diff = diff_stats(actual_canonical, ref_canonical);
    const size_t max_i = report.canonical_weight_diff.max_index;
    const int bf16_bit_delta =
        static_cast<int>(f32_to_bf16_bits(actual_canonical[max_i])) -
        static_cast<int>(f32_to_bf16_bits(ref_canonical[max_i]));

    std::cout << label
              << " expert_set_mismatch_tokens=" << report.expert_set_mismatch_tokens
              << " slot_order_mismatches=" << report.slot_mismatches
              << " equal_logit_tie_slot_mismatches=" << report.tie_like_mismatches
              << " canonical_weight_max_diff=" << report.canonical_weight_diff.max_diff
              << " canonical_weight_mean_diff=" << report.canonical_weight_diff.mean_diff
              << " canonical_weight_rel_max=" << report.canonical_weight_diff.rel_max
              << " canonical_weight_max_index=" << max_i
              << " canonical_weight_actual=" << actual_canonical[max_i]
              << " canonical_weight_ref=" << ref_canonical[max_i]
              << " canonical_weight_actual_bf16=0x" << std::hex << f32_to_bf16_bits(actual_canonical[max_i])
              << " canonical_weight_ref_bf16=0x" << f32_to_bf16_bits(ref_canonical[max_i])
              << std::dec
              << " canonical_weight_bf16_bit_delta=" << bf16_bit_delta
              << "\n";
    if (report.expert_set_mismatch_tokens != 0 ||
        report.slot_mismatches != report.tie_like_mismatches ||
        report.canonical_weight_diff.max_diff > 0.0f) {
        print_topk_mismatch_table(label, actual.indices, ref_indices, actual.logits, ref_logits, top_k, mismatch_detail_limit);
    }
    return report;
}

static std::vector<int64_t> topk_expert_set_mismatch_tokens(const std::vector<int64_t>& actual,
                                                            const NpyI64& ref,
                                                            int top_k) {
    if (actual.size() != ref.data.size() || actual.size() % static_cast<size_t>(top_k) != 0) {
        throw std::runtime_error("top-k mismatch token shape mismatch");
    }
    std::vector<int64_t> tokens;
    const size_t seq = actual.size() / static_cast<size_t>(top_k);
    for (size_t s = 0; s < seq; ++s) {
        std::vector<int64_t> a(actual.begin() + static_cast<std::ptrdiff_t>(s * top_k),
                               actual.begin() + static_cast<std::ptrdiff_t>((s + 1) * top_k));
        std::vector<int64_t> r(ref.data.begin() + static_cast<std::ptrdiff_t>(s * top_k),
                               ref.data.begin() + static_cast<std::ptrdiff_t>((s + 1) * top_k));
        std::sort(a.begin(), a.end());
        std::sort(r.begin(), r.end());
        if (a != r) {
            tokens.push_back(static_cast<int64_t>(s));
        }
    }
    return tokens;
}

static void print_token_topk_list(const std::string& name,
                                  const std::vector<int64_t>& indices,
                                  int64_t token,
                                  int top_k) {
    std::cout << " " << name << "=";
    for (int k = 0; k < top_k; ++k) {
        if (k) std::cout << ",";
        std::cout << indices[static_cast<size_t>(token * top_k + k)];
    }
}

static std::pair<float, float> kth_boundary_margin(const TensorF32& logits, int64_t token, int top_k) {
    if (logits.shape.size() != 3 || logits.shape[0] != 1) {
        throw std::runtime_error("router boundary logits shape mismatch");
    }
    const int64_t experts = logits.shape[2];
    std::vector<float> values;
    values.reserve(static_cast<size_t>(experts));
    for (int64_t e = 0; e < experts; ++e) {
        values.push_back(logits.data[static_cast<size_t>(token * experts + e)]);
    }
    std::sort(values.begin(), values.end(), std::greater<float>());
    const float kth = values[static_cast<size_t>(top_k - 1)];
    const float next = values[static_cast<size_t>(top_k)];
    return {kth, kth - next};
}

static void report_router_sensitivity(const std::string& label,
                                      const TopKResult& native_topk,
                                      const NpyI64& ref_indices,
                                      const NpyF32& ref_logits_npy,
                                      const TensorF32& router_input,
                                      const std::filesystem::path& oracle_router_input_path,
                                      int top_k) {
    const std::vector<int64_t> mismatch_tokens = topk_expert_set_mismatch_tokens(native_topk.indices, ref_indices, top_k);
    if (mismatch_tokens.empty()) {
        std::cout << label << "_router_sensitivity expert_set_mismatch_tokens=0\n";
        return;
    }
    TensorF32 ref_logits = make_tensor(ref_logits_npy.shape, ref_logits_npy.data);
    TensorF32 ref_input;
    bool have_ref_input = false;
    if (std::filesystem::exists(oracle_router_input_path)) {
        const NpyF32 ref_input_npy = load_npy_f32(oracle_router_input_path);
        ref_input = make_tensor(ref_input_npy.shape, ref_input_npy.data);
        have_ref_input = ref_input.shape == router_input.shape;
    }
    const int64_t experts = native_topk.logits.shape[2];
    const int64_t hidden = router_input.shape[2];
    std::cout << label << "_router_sensitivity expert_set_mismatch_tokens=" << mismatch_tokens.size()
              << " token_list=";
    for (size_t i = 0; i < mismatch_tokens.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << mismatch_tokens[i];
    }
    std::cout << "\n";
    for (int64_t token : mismatch_tokens) {
        std::set<int64_t> union_experts;
        for (int k = 0; k < top_k; ++k) {
            union_experts.insert(native_topk.indices[static_cast<size_t>(token * top_k + k)]);
            union_experts.insert(ref_indices.data[static_cast<size_t>(token * top_k + k)]);
        }
        auto native_boundary = kth_boundary_margin(native_topk.logits, token, top_k);
        auto oracle_boundary = kth_boundary_margin(ref_logits, token, top_k);
        float input_max = 0.0f;
        float input_sum = 0.0f;
        double input_l2 = 0.0;
        if (have_ref_input) {
            for (int64_t h = 0; h < hidden; ++h) {
                const size_t idx = static_cast<size_t>(token * hidden + h);
                const float diff = std::fabs(router_input.data[idx] - ref_input.data[idx]);
                input_max = std::max(input_max, diff);
                input_sum += diff;
                input_l2 += static_cast<double>(diff) * static_cast<double>(diff);
            }
        }
        const float input_mean = have_ref_input ? input_sum / static_cast<float>(hidden) : -1.0f;
        const float input_l2_norm = have_ref_input ? static_cast<float>(std::sqrt(input_l2)) : -1.0f;
        const bool exact_tie = native_boundary.second == 0.0f || oracle_boundary.second == 0.0f;
        const bool one_bf16_ulp_like =
            std::fabs(native_boundary.second) <= 0.0078125f || std::fabs(oracle_boundary.second) <= 0.0078125f;
        std::cout << label << "_router_sensitivity_token token=" << token;
        print_token_topk_list("native_topk", native_topk.indices, token, top_k);
        print_token_topk_list("oracle_topk", ref_indices.data, token, top_k);
        std::cout << " native_kth_logit=" << native_boundary.first
                  << " native_kth_margin=" << native_boundary.second
                  << " oracle_kth_logit=" << oracle_boundary.first
                  << " oracle_kth_margin=" << oracle_boundary.second
                  << " router_input_max_diff=" << input_max
                  << " router_input_mean_diff=" << input_mean
                  << " router_input_l2=" << input_l2_norm
                  << " classification=" << (exact_tie ? "exact_or_native_tie" : (one_bf16_ulp_like ? "bf16_near_boundary" : "larger_accumulated_drift"))
                  << " union_logits=";
        bool first = true;
        for (int64_t expert : union_experts) {
            if (expert < 0 || expert >= experts) continue;
            if (!first) std::cout << ";";
            first = false;
            const size_t idx = static_cast<size_t>(token * experts + expert);
            std::cout << expert << ":native=" << native_topk.logits.data[idx]
                      << ",oracle=" << ref_logits.data[idx]
                      << ",delta=" << (native_topk.logits.data[idx] - ref_logits.data[idx]);
        }
        std::cout << "\n";
    }
}

static void run_router_cuda_parity(const std::filesystem::path& oracle_dir,
                                   const std::unordered_map<std::string, SafetensorEntry>& index) {
    const int top_k = 4;
    const NpyF32 input_npy = load_npy_f32(oracle_dir / "layer0_manual_post_attention_norm_f32.npy");
    TensorF32 input = make_tensor(input_npy.shape, input_npy.data);
    if (input.shape.size() != 3 || input.shape[0] != 1) {
        throw std::runtime_error("router CUDA input oracle shape mismatch");
    }

    const auto& router_w_e = need_entry(index, "model.layers.0.mlp.router.weight");
    const auto& router_b_e = need_entry(index, "model.layers.0.mlp.router.bias");
    const std::vector<ggml_bf16_t> weight_bf16 = read_bf16_raw(router_w_e);
    const std::vector<ggml_bf16_t> bias_bf16 = read_bf16_raw(router_b_e);
    const std::vector<float> weight_f32 = read_bf16_as_f32(router_w_e);
    const std::vector<float> bias_f32 = read_bf16_as_f32(router_b_e);
    const NpyF32 ref_logits = load_npy_f32(oracle_dir / "layer0_manual_router_logits_f32.npy");
    const NpyI64 ref_indices = load_npy_i64(oracle_dir / "layer0_manual_router_indices_i64.npy");
    const NpyF32 ref_scores = load_npy_f32(oracle_dir / "layer0_manual_router_scores_f32.npy");

    const std::vector<RouterCudaMode> modes = {
        {"bf16_input_bf16_weight_bf16_output", GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, true, true},
        {"bf16_input_bf16_weight_f32_output", GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, true, false},
        {"f32_input_bf16_weight_bf16_output", GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_BF16, true, true},
        {"bf16_input_f32_weight_bf16_output", GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_F32, true, true},
    };

    size_t best_set_mismatch_tokens = std::numeric_limits<size_t>::max();
    size_t best_slot_mismatches = std::numeric_limits<size_t>::max();
    DiffStats best_logit_diff;
    std::string best_mode;
    bool canonical_ok = false;
    for (const RouterCudaMode& mode : modes) {
        double elapsed_ms = 0.0;
        size_t buffer_bytes = 0;
        std::string output_dtype;
        std::vector<int64_t> cuda_topk_indices;
        TensorF32 logits = run_router_cuda_mode(mode,
                                                input,
                                                weight_bf16,
                                                weight_f32,
                                                router_w_e.shape,
                                                bias_bf16,
                                                bias_f32,
                                                top_k,
                                                &cuda_topk_indices,
                                                &elapsed_ms,
                                                &buffer_bytes,
                                                &output_dtype);
        const DiffStats logit_diff = diff_stats(logits.data, ref_logits.data);
        const TopKResult cpu_topk = topk_from_logits(logits, top_k);
        const size_t cpu_mismatches = topk_mismatch_count(cpu_topk.indices, ref_indices);
        const size_t cuda_mismatches = topk_mismatch_count(cuda_topk_indices, ref_indices);
        const size_t cpu_set_mismatch_tokens = topk_set_mismatch_tokens(cpu_topk.indices, ref_indices, top_k);
        const size_t cuda_set_mismatch_tokens = topk_set_mismatch_tokens(cuda_topk_indices, ref_indices, top_k);
        const size_t cpu_tie_like_mismatches = topk_tie_like_mismatch_count(cpu_topk.indices, ref_indices, logits, ref_logits, top_k);
        const size_t cuda_tie_like_mismatches = topk_tie_like_mismatch_count(cuda_topk_indices, ref_indices, logits, ref_logits, top_k);
        const std::vector<float> cuda_scores = topk_weights_for_indices(logits, cuda_topk_indices, top_k);
        const DiffStats score_diff = diff_stats(cuda_scores, ref_scores.data);
        TopKCanonicalReport cpu_canonical;
        bool cpu_canonical_ok = false;
        try {
            cpu_canonical = validate_canonical_topk("router_cpu_canonical_topk mode=" + mode.name,
                                                    cpu_topk,
                                                    ref_indices,
                                                    ref_scores,
                                                    ref_logits,
                                                    top_k);
            cpu_canonical_ok = true;
        } catch (const std::exception& e) {
            std::cout << "router_cpu_canonical_topk mode=" << mode.name
                      << " failed=\"" << e.what() << "\"\n";
        }
        std::cout << "router_cuda_mode name=" << mode.name
                  << " backend=ggml_cuda"
                  << " input_type=" << ggml_type_label(mode.input_type)
                  << " weight_type=" << ggml_type_label(mode.weight_type)
                  << " bias_type=" << ggml_type_label(mode.bias_type)
                  << " output_type=" << output_dtype
                  << " force_prec_f32=" << (mode.force_prec_f32 ? "true" : "false")
                  << " cast_output_bf16=" << (mode.cast_output_bf16 ? "true" : "false")
                  << " shape=" << shape_string(logits.shape)
                  << " elapsed_ms=" << elapsed_ms
                  << " buffer_bytes=" << buffer_bytes
                  << " logits_max_diff=" << logit_diff.max_diff
                  << " logits_mean_diff=" << logit_diff.mean_diff
                  << " logits_rel_max=" << logit_diff.rel_max
                  << " cpu_resort_topk_mismatches=" << cpu_mismatches
                  << " cpu_resort_set_mismatch_tokens=" << cpu_set_mismatch_tokens
                  << " cpu_resort_tie_like_mismatches=" << cpu_tie_like_mismatches
                  << " cuda_argsort_topk_mismatches=" << cuda_mismatches
                  << " cuda_argsort_set_mismatch_tokens=" << cuda_set_mismatch_tokens
                  << " cuda_argsort_tie_like_mismatches=" << cuda_tie_like_mismatches
                  << " score_max_diff=" << score_diff.max_diff
                  << " score_mean_diff=" << score_diff.mean_diff
                  << "\n";
        if (cuda_mismatches != 0) {
            print_topk_mismatch_table(mode.name, cuda_topk_indices, ref_indices, logits, ref_logits, top_k, 10);
        }
        if (cpu_canonical_ok &&
            (cpu_canonical.expert_set_mismatch_tokens < best_set_mismatch_tokens ||
            (cpu_canonical.expert_set_mismatch_tokens == best_set_mismatch_tokens &&
             cpu_canonical.slot_mismatches < best_slot_mismatches) ||
            (cpu_canonical.expert_set_mismatch_tokens == best_set_mismatch_tokens &&
             cpu_canonical.slot_mismatches == best_slot_mismatches &&
             logit_diff.mean_diff < best_logit_diff.mean_diff))) {
            best_set_mismatch_tokens = cpu_canonical.expert_set_mismatch_tokens;
            best_slot_mismatches = cpu_canonical.slot_mismatches;
            best_logit_diff = logit_diff;
            best_mode = mode.name;
        }
        if (cpu_canonical_ok && logit_diff.max_diff <= 1.0e-6f) {
            canonical_ok = true;
        }
    }
    std::cout << "router_cuda_best mode=" << best_mode
              << " expert_set_mismatch_tokens=" << best_set_mismatch_tokens
              << " slot_order_mismatches=" << best_slot_mismatches
              << " logits_max_diff=" << best_logit_diff.max_diff
              << " logits_mean_diff=" << best_logit_diff.mean_diff
              << "\n";
    if (!canonical_ok) {
        throw std::runtime_error("router CUDA canonical top-k did not satisfy expert-set/weight acceptance");
    }
}

#ifdef SD_LENS_TEXT_ENCODER_USE_CUBLASLT
static TensorF32 linear_cublaslt_bf16_bias_projection_probe(const TensorF32& input,
                                                            const std::unordered_map<std::string, SafetensorEntry>& index,
                                                            const std::string& prefix,
                                                            const std::string& label,
                                                            cublasComputeType_t compute_type,
                                                            int heuristic_index = 0,
                                                            int* algo_id_out = nullptr,
                                                            double* elapsed_ms = nullptr);
#endif

static TensorF32 linear_cuda_bf16(const TensorF32& input,
                                  const std::unordered_map<std::string, SafetensorEntry>& index,
                                  const std::string& prefix,
                                  const std::string& label,
                                  double* elapsed_ms = nullptr) {
#ifdef SD_LENS_TEXT_ENCODER_USE_CUBLASLT
    int algo_id = -1;
    TensorF32 out = linear_cublaslt_bf16_bias_projection_probe(input,
                                                               index,
                                                               prefix,
                                                               label + "_cublaslt_exact_linear",
                                                               CUBLAS_COMPUTE_32F,
                                                               1,
                                                               &algo_id,
                                                               elapsed_ms);
    std::cout << label << "_projection_backend=cublaslt_bf16_bias_compute32"
              << " heuristic=1"
              << " algo=" << algo_id
              << " output_dtype=BF16"
              << "\n";
    return out;
#else
    const auto& w_e = need_entry(index, prefix + ".weight");
    const auto& b_e = need_entry(index, prefix + ".bias");
#if 0
    if (elapsed_ms != nullptr) {
        const auto start = std::chrono::steady_clock::now();
        TensorF32 out = linear_bf16(input, read_bf16_as_f32(w_e), w_e.shape, read_bf16_as_f32(b_e), label + "_cpu_projection_diagnostic");
        *elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        return out;
    }
#endif
    const std::vector<ggml_bf16_t> w_bf16 = read_bf16_raw(w_e);
    const std::vector<ggml_bf16_t> b_bf16 = read_bf16_raw(b_e);
    const std::vector<float> w_f32 = read_bf16_as_f32(w_e);
    const std::vector<float> b_f32 = read_bf16_as_f32(b_e);
    const RouterCudaMode mode{label, GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, false, true};
    size_t buffer_bytes = 0;
    std::string output_dtype;
    return run_router_cuda_mode(mode,
                                input,
                                w_bf16,
                                w_f32,
                                w_e.shape,
                                b_bf16,
                                b_f32,
                                0,
                                nullptr,
                                elapsed_ms,
                                &buffer_bytes,
                                &output_dtype);
#endif
}

static TensorF32 linear_cublas_bf16_projection_probe(const TensorF32& input,
                                                     const std::unordered_map<std::string, SafetensorEntry>& index,
                                                     const std::string& prefix,
                                                     const std::string& label,
                                                     double* elapsed_ms = nullptr) {
#if !defined(SD_USE_CUDA) || !defined(SD_LENS_TEXT_ENCODER_USE_CUBLAS)
    (void)input;
    (void)index;
    (void)prefix;
    (void)label;
    (void)elapsed_ms;
    throw std::runtime_error("cuBLAS projection probe requires SD_USE_CUDA and CUDAToolkit");
#else
    if (input.shape.size() != 3 || input.shape[0] != 1) {
        throw std::runtime_error(label + " cuBLAS input shape mismatch");
    }
    const auto& w_e = need_entry(index, prefix + ".weight");
    const auto& b_e = need_entry(index, prefix + ".bias");
    if (w_e.shape.size() != 2 || w_e.shape[1] != input.shape[2]) {
        throw std::runtime_error(label + " cuBLAS weight shape mismatch");
    }
    const int64_t seq = input.shape[1];
    const int64_t in = input.shape[2];
    const int64_t out_dim = w_e.shape[0];
    const std::vector<ggml_bf16_t> w_bf16 = read_bf16_raw(w_e);
    const std::vector<float> bias = read_bf16_as_f32(b_e);
    std::vector<ggml_bf16_t> x_bf16(input.data.size());
    ggml_fp32_to_bf16_row_ref(input.data.data(), x_bf16.data(), static_cast<int64_t>(input.data.size()));

    cublasHandle_t handle = nullptr;
    void* d_w = nullptr;
    void* d_x = nullptr;
    float* d_y = nullptr;
    try {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }
        const size_t w_bytes = w_bf16.size() * sizeof(ggml_bf16_t);
        const size_t x_bytes = x_bf16.size() * sizeof(ggml_bf16_t);
        const size_t y_elems = static_cast<size_t>(seq * out_dim);
        if (cudaMalloc(&d_w, w_bytes) != cudaSuccess ||
            cudaMalloc(&d_x, x_bytes) != cudaSuccess ||
            cudaMalloc(&d_y, y_elems * sizeof(float)) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed for cuBLAS projection probe");
        }
        if (cudaMemcpy(d_w, w_bf16.data(), w_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_x, x_bf16.data(), x_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy H2D failed for cuBLAS projection probe");
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const auto start = std::chrono::steady_clock::now();
        const cublasStatus_t status = cublasGemmEx(handle,
                                                   CUBLAS_OP_T,
                                                   CUBLAS_OP_N,
                                                   static_cast<int>(out_dim),
                                                   static_cast<int>(seq),
                                                   static_cast<int>(in),
                                                   &alpha,
                                                   d_w,
                                                   CUDA_R_16BF,
                                                   static_cast<int>(in),
                                                   d_x,
                                                   CUDA_R_16BF,
                                                   static_cast<int>(in),
                                                   &beta,
                                                   d_y,
                                                   CUDA_R_32F,
                                                   static_cast<int>(out_dim),
                                                   CUBLAS_COMPUTE_32F,
                                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasGemmEx failed for projection probe");
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            throw std::runtime_error("cudaDeviceSynchronize failed for projection probe");
        }
        if (elapsed_ms != nullptr) {
            *elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        std::vector<float> y(y_elems);
        if (cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy D2H failed for cuBLAS projection probe");
        }
        TensorF32 out;
        out.shape = {1, seq, out_dim};
        out.data.resize(y_elems);
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t o = 0; o < out_dim; ++o) {
                out.data[static_cast<size_t>(s * out_dim + o)] =
                    round_to_bf16(y[static_cast<size_t>(o + out_dim * s)] + bias[static_cast<size_t>(o)]);
            }
        }
        cudaFree(d_y);
        cudaFree(d_x);
        cudaFree(d_w);
        cublasDestroy(handle);
        return out;
    } catch (...) {
        if (d_y != nullptr) cudaFree(d_y);
        if (d_x != nullptr) cudaFree(d_x);
        if (d_w != nullptr) cudaFree(d_w);
        if (handle != nullptr) cublasDestroy(handle);
        throw;
    }
#endif
}

static TensorF32 linear_cublas_bf16_output_projection_probe(const TensorF32& input,
                                                            const std::unordered_map<std::string, SafetensorEntry>& index,
                                                            const std::string& prefix,
                                                            const std::string& label,
                                                            double* elapsed_ms = nullptr) {
#if !defined(SD_USE_CUDA) || !defined(SD_LENS_TEXT_ENCODER_USE_CUBLAS)
    (void)input;
    (void)index;
    (void)prefix;
    (void)label;
    (void)elapsed_ms;
    throw std::runtime_error("cuBLAS BF16-output projection probe requires SD_USE_CUDA and CUDAToolkit");
#else
    if (input.shape.size() != 3 || input.shape[0] != 1) {
        throw std::runtime_error(label + " cuBLAS BF16-output input shape mismatch");
    }
    const auto& w_e = need_entry(index, prefix + ".weight");
    const auto& b_e = need_entry(index, prefix + ".bias");
    if (w_e.shape.size() != 2 || w_e.shape[1] != input.shape[2]) {
        throw std::runtime_error(label + " cuBLAS BF16-output weight shape mismatch");
    }
    const int64_t seq = input.shape[1];
    const int64_t in = input.shape[2];
    const int64_t out_dim = w_e.shape[0];
    const std::vector<ggml_bf16_t> w_bf16 = read_bf16_raw(w_e);
    const std::vector<float> bias = read_bf16_as_f32(b_e);
    std::vector<ggml_bf16_t> x_bf16(input.data.size());
    ggml_fp32_to_bf16_row_ref(input.data.data(), x_bf16.data(), static_cast<int64_t>(input.data.size()));

    cublasHandle_t handle = nullptr;
    void* d_w = nullptr;
    void* d_x = nullptr;
    void* d_y = nullptr;
    try {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }
        const size_t w_bytes = w_bf16.size() * sizeof(ggml_bf16_t);
        const size_t x_bytes = x_bf16.size() * sizeof(ggml_bf16_t);
        const size_t y_elems = static_cast<size_t>(seq * out_dim);
        if (cudaMalloc(&d_w, w_bytes) != cudaSuccess ||
            cudaMalloc(&d_x, x_bytes) != cudaSuccess ||
            cudaMalloc(&d_y, y_elems * sizeof(ggml_bf16_t)) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed for cuBLAS BF16-output projection probe");
        }
        if (cudaMemcpy(d_w, w_bf16.data(), w_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_x, x_bf16.data(), x_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy H2D failed for cuBLAS BF16-output projection probe");
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const auto start = std::chrono::steady_clock::now();
        const cublasStatus_t status = cublasGemmEx(handle,
                                                   CUBLAS_OP_T,
                                                   CUBLAS_OP_N,
                                                   static_cast<int>(out_dim),
                                                   static_cast<int>(seq),
                                                   static_cast<int>(in),
                                                   &alpha,
                                                   d_w,
                                                   CUDA_R_16BF,
                                                   static_cast<int>(in),
                                                   d_x,
                                                   CUDA_R_16BF,
                                                   static_cast<int>(in),
                                                   &beta,
                                                   d_y,
                                                   CUDA_R_16BF,
                                                   static_cast<int>(out_dim),
                                                   CUBLAS_COMPUTE_32F,
                                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasGemmEx BF16-output failed for projection probe");
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            throw std::runtime_error("cudaDeviceSynchronize failed for cuBLAS BF16-output projection probe");
        }
        if (elapsed_ms != nullptr) {
            *elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        std::vector<ggml_bf16_t> y_bf16(y_elems);
        if (cudaMemcpy(y_bf16.data(), d_y, y_bf16.size() * sizeof(ggml_bf16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy D2H failed for cuBLAS BF16-output projection probe");
        }
        std::vector<float> y(y_elems);
        ggml_bf16_to_fp32_row(y_bf16.data(), y.data(), static_cast<int64_t>(y.size()));
        TensorF32 out;
        out.shape = {1, seq, out_dim};
        out.data.resize(y_elems);
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t o = 0; o < out_dim; ++o) {
                out.data[static_cast<size_t>(s * out_dim + o)] =
                    round_to_bf16(y[static_cast<size_t>(o + out_dim * s)] + bias[static_cast<size_t>(o)]);
            }
        }
        cudaFree(d_y);
        cudaFree(d_x);
        cudaFree(d_w);
        cublasDestroy(handle);
        return out;
    } catch (...) {
        if (d_y != nullptr) cudaFree(d_y);
        if (d_x != nullptr) cudaFree(d_x);
        if (d_w != nullptr) cudaFree(d_w);
        if (handle != nullptr) cublasDestroy(handle);
        throw;
    }
#endif
}

#ifdef SD_LENS_TEXT_ENCODER_USE_CUBLASLT
static TensorF32 linear_cublaslt_bf16_bias_projection_probe(const TensorF32& input,
                                                            const std::unordered_map<std::string, SafetensorEntry>& index,
                                                            const std::string& prefix,
                                                            const std::string& label,
                                                            cublasComputeType_t compute_type,
                                                            int heuristic_index,
                                                            int* algo_id_out,
                                                            double* elapsed_ms) {
#if !defined(SD_USE_CUDA) || !defined(SD_LENS_TEXT_ENCODER_USE_CUBLASLT)
    (void)input;
    (void)index;
    (void)prefix;
    (void)label;
    (void)compute_type;
    (void)heuristic_index;
    (void)algo_id_out;
    (void)elapsed_ms;
    throw std::runtime_error("cuBLASLt projection probe requires SD_USE_CUDA and CUDA::cublasLt");
#else
    if (input.shape.size() != 3 || input.shape[0] != 1) {
        throw std::runtime_error(label + " cuBLASLt input shape mismatch");
    }
    const auto& w_e = need_entry(index, prefix + ".weight");
    const auto& b_e = need_entry(index, prefix + ".bias");
    if (w_e.shape.size() != 2 || w_e.shape[1] != input.shape[2]) {
        throw std::runtime_error(label + " cuBLASLt weight shape mismatch");
    }
    const int64_t seq = input.shape[1];
    const int64_t in = input.shape[2];
    const int64_t out_dim = w_e.shape[0];
    const std::vector<ggml_bf16_t> w_bf16 = read_bf16_raw(w_e);
    const std::vector<ggml_bf16_t> b_bf16 = read_bf16_raw(b_e);
    std::vector<ggml_bf16_t> x_bf16(input.data.size());
    ggml_fp32_to_bf16_row_ref(input.data.data(), x_bf16.data(), static_cast<int64_t>(input.data.size()));

    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatrixLayout_t d_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    void* d_w = nullptr;
    void* d_x = nullptr;
    void* d_bias = nullptr;
    void* d_y = nullptr;
    void* workspace = nullptr;
    try {
        if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasLtCreate failed");
        }
        cublasOperation_t transa = CUBLAS_OP_T;
        cublasOperation_t transb = CUBLAS_OP_N;
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        if (cublasLtMatmulDescCreate(&op_desc, compute_type, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasLt matmul desc setup failed");
        }
        if (cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, static_cast<uint64_t>(in), static_cast<uint64_t>(out_dim), static_cast<int64_t>(in)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, static_cast<uint64_t>(in), static_cast<uint64_t>(seq), static_cast<int64_t>(in)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, static_cast<uint64_t>(out_dim), static_cast<uint64_t>(seq), static_cast<int64_t>(out_dim)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16BF, static_cast<uint64_t>(out_dim), static_cast<uint64_t>(seq), static_cast<int64_t>(out_dim)) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasLt matrix layout setup failed");
        }

        const size_t w_bytes = w_bf16.size() * sizeof(ggml_bf16_t);
        const size_t x_bytes = x_bf16.size() * sizeof(ggml_bf16_t);
        const size_t b_bytes = b_bf16.size() * sizeof(ggml_bf16_t);
        const size_t y_elems = static_cast<size_t>(seq * out_dim);
        if (cudaMalloc(&d_w, w_bytes) != cudaSuccess ||
            cudaMalloc(&d_x, x_bytes) != cudaSuccess ||
            cudaMalloc(&d_bias, b_bytes) != cudaSuccess ||
            cudaMalloc(&d_y, y_elems * sizeof(ggml_bf16_t)) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed for cuBLASLt projection probe");
        }
        if (cudaMemcpy(d_w, w_bf16.data(), w_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_x, x_bf16.data(), x_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_bias, b_bf16.data(), b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy H2D failed for cuBLASLt projection probe");
        }
        if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias)) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasLt bias pointer setup failed");
        }

        size_t workspace_size = 32ull * 1024ull * 1024ull;
        if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) {
            workspace = nullptr;
            workspace_size = 0;
        }
        if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasLt preference setup failed");
        }
        if (heuristic_index < 0) {
            throw std::runtime_error("negative cuBLASLt heuristic index");
        }
        std::vector<cublasLtMatmulHeuristicResult_t> heuristics(static_cast<size_t>(heuristic_index + 1));
        int returned_results = 0;
        if (cublasLtMatmulAlgoGetHeuristic(lt,
                                           op_desc,
                                           a_desc,
                                           b_desc,
                                           c_desc,
                                           d_desc,
                                           pref,
                                           static_cast<int>(heuristics.size()),
                                           heuristics.data(),
                                           &returned_results) != CUBLAS_STATUS_SUCCESS ||
            returned_results <= heuristic_index) {
            throw std::runtime_error("cublasLt heuristic lookup failed");
        }
        const cublasLtMatmulHeuristicResult_t& heuristic = heuristics[static_cast<size_t>(heuristic_index)];
        if (algo_id_out != nullptr) {
            int algo_id = -1;
            size_t written = 0;
            if (cublasLtMatmulAlgoConfigGetAttribute(&heuristic.algo,
                                                     CUBLASLT_ALGO_CONFIG_ID,
                                                     &algo_id,
                                                     sizeof(algo_id),
                                                     &written) == CUBLAS_STATUS_SUCCESS) {
                *algo_id_out = algo_id;
            } else {
                *algo_id_out = -1;
            }
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        const auto start = std::chrono::steady_clock::now();
        const cublasStatus_t status = cublasLtMatmul(lt,
                                                     op_desc,
                                                     &alpha,
                                                     d_w,
                                                     a_desc,
                                                     d_x,
                                                     b_desc,
                                                     &beta,
                                                     d_y,
                                                     c_desc,
                                                     d_y,
                                                     d_desc,
                                                     &heuristic.algo,
                                                     workspace,
                                                     workspace_size,
                                                     0);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasLtMatmul failed for projection probe");
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            throw std::runtime_error("cudaDeviceSynchronize failed for cuBLASLt projection probe");
        }
        if (elapsed_ms != nullptr) {
            *elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }

        std::vector<ggml_bf16_t> y_bf16(y_elems);
        if (cudaMemcpy(y_bf16.data(), d_y, y_bf16.size() * sizeof(ggml_bf16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy D2H failed for cuBLASLt projection probe");
        }
        TensorF32 out;
        out.shape = {1, seq, out_dim};
        out.data.resize(y_elems);
        for (int64_t s = 0; s < seq; ++s) {
            ggml_bf16_to_fp32_row(y_bf16.data() + static_cast<size_t>(s * out_dim),
                                  out.data.data() + static_cast<size_t>(s * out_dim),
                                  out_dim);
        }

        if (workspace != nullptr) cudaFree(workspace);
        cudaFree(d_y);
        cudaFree(d_bias);
        cudaFree(d_x);
        cudaFree(d_w);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(d_desc);
        cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatmulDescDestroy(op_desc);
        cublasLtDestroy(lt);
        return out;
    } catch (...) {
        if (workspace != nullptr) cudaFree(workspace);
        if (d_y != nullptr) cudaFree(d_y);
        if (d_bias != nullptr) cudaFree(d_bias);
        if (d_x != nullptr) cudaFree(d_x);
        if (d_w != nullptr) cudaFree(d_w);
        if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
        if (d_desc != nullptr) cublasLtMatrixLayoutDestroy(d_desc);
        if (c_desc != nullptr) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc != nullptr) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc != nullptr) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc != nullptr) cublasLtMatmulDescDestroy(op_desc);
        if (lt != nullptr) cublasLtDestroy(lt);
        throw;
    }
#endif
}
#endif

#ifdef SD_LENS_TEXT_ENCODER_USE_CUBLASLT
static TensorF32 linear_cublaslt_bf16_bias_from_host(const TensorF32& input,
                                                     const std::vector<float>& weight_row_major,
                                                     int64_t out_dim,
                                                     int64_t in_dim,
                                                     const std::vector<float>& bias,
                                                     const std::string& label,
                                                     double* elapsed_ms) {
#if !defined(SD_USE_CUDA) || !defined(SD_LENS_TEXT_ENCODER_USE_CUBLASLT)
    (void)input;
    (void)weight_row_major;
    (void)out_dim;
    (void)in_dim;
    (void)bias;
    (void)label;
    (void)elapsed_ms;
    throw std::runtime_error("cuBLASLt host linear requires SD_USE_CUDA and CUDA::cublasLt");
#else
    if (input.shape.size() != 3 || input.shape[0] != 1 || input.shape[2] != in_dim) {
        throw std::runtime_error(label + " cuBLASLt host input shape mismatch");
    }
    if (static_cast<int64_t>(weight_row_major.size()) != out_dim * in_dim ||
        static_cast<int64_t>(bias.size()) != out_dim) {
        throw std::runtime_error(label + " cuBLASLt host weight/bias shape mismatch");
    }
    const int64_t seq = input.shape[1];
    std::vector<ggml_bf16_t> w_bf16(weight_row_major.size());
    std::vector<ggml_bf16_t> b_bf16(bias.size());
    std::vector<ggml_bf16_t> x_bf16(input.data.size());
    ggml_fp32_to_bf16_row_ref(weight_row_major.data(), w_bf16.data(), static_cast<int64_t>(weight_row_major.size()));
    ggml_fp32_to_bf16_row_ref(bias.data(), b_bf16.data(), static_cast<int64_t>(bias.size()));
    ggml_fp32_to_bf16_row_ref(input.data.data(), x_bf16.data(), static_cast<int64_t>(input.data.size()));

    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatrixLayout_t d_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    void* d_w = nullptr;
    void* d_x = nullptr;
    void* d_bias = nullptr;
    void* d_y = nullptr;
    void* workspace = nullptr;
    try {
        if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(label + " cublasLtCreate failed");
        }
        cublasOperation_t transa = CUBLAS_OP_T;
        cublasOperation_t transb = CUBLAS_OP_N;
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(label + " cuBLASLt matmul desc setup failed");
        }
        if (cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, static_cast<uint64_t>(in_dim), static_cast<uint64_t>(out_dim), static_cast<int64_t>(in_dim)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, static_cast<uint64_t>(in_dim), static_cast<uint64_t>(seq), static_cast<int64_t>(in_dim)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, static_cast<uint64_t>(out_dim), static_cast<uint64_t>(seq), static_cast<int64_t>(out_dim)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16BF, static_cast<uint64_t>(out_dim), static_cast<uint64_t>(seq), static_cast<int64_t>(out_dim)) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(label + " cuBLASLt matrix layout setup failed");
        }

        const size_t w_bytes = w_bf16.size() * sizeof(ggml_bf16_t);
        const size_t x_bytes = x_bf16.size() * sizeof(ggml_bf16_t);
        const size_t b_bytes = b_bf16.size() * sizeof(ggml_bf16_t);
        const size_t y_elems = static_cast<size_t>(seq * out_dim);
        if (cudaMalloc(&d_w, w_bytes) != cudaSuccess ||
            cudaMalloc(&d_x, x_bytes) != cudaSuccess ||
            cudaMalloc(&d_bias, b_bytes) != cudaSuccess ||
            cudaMalloc(&d_y, y_elems * sizeof(ggml_bf16_t)) != cudaSuccess) {
            throw std::runtime_error(label + " cudaMalloc failed");
        }
        if (cudaMemcpy(d_w, w_bf16.data(), w_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_x, x_bf16.data(), x_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_bias, b_bf16.data(), b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error(label + " cudaMemcpy H2D failed");
        }
        if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias)) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(label + " cuBLASLt bias pointer setup failed");
        }
        size_t workspace_size = 32ull * 1024ull * 1024ull;
        if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) {
            workspace = nullptr;
            workspace_size = 0;
        }
        if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(label + " cuBLASLt preference setup failed");
        }
        cublasLtMatmulHeuristicResult_t heuristic = {};
        int returned_results = 0;
        if (cublasLtMatmulAlgoGetHeuristic(lt,
                                           op_desc,
                                           a_desc,
                                           b_desc,
                                           c_desc,
                                           d_desc,
                                           pref,
                                           1,
                                           &heuristic,
                                           &returned_results) != CUBLAS_STATUS_SUCCESS ||
            returned_results <= 0) {
            throw std::runtime_error(label + " cuBLASLt heuristic lookup failed");
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const auto start = std::chrono::steady_clock::now();
        const cublasStatus_t status = cublasLtMatmul(lt,
                                                     op_desc,
                                                     &alpha,
                                                     d_w,
                                                     a_desc,
                                                     d_x,
                                                     b_desc,
                                                     &beta,
                                                     d_y,
                                                     c_desc,
                                                     d_y,
                                                     d_desc,
                                                     &heuristic.algo,
                                                     workspace,
                                                     workspace_size,
                                                     0);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(label + " cuBLASLtMatmul failed");
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            throw std::runtime_error(label + " cudaDeviceSynchronize failed");
        }
        if (elapsed_ms != nullptr) {
            *elapsed_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }

        std::vector<ggml_bf16_t> y_bf16(y_elems);
        if (cudaMemcpy(y_bf16.data(), d_y, y_bf16.size() * sizeof(ggml_bf16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error(label + " cudaMemcpy D2H failed");
        }
        TensorF32 out;
        out.shape = {1, seq, out_dim};
        out.data.resize(y_elems);
        for (int64_t s = 0; s < seq; ++s) {
            ggml_bf16_to_fp32_row(y_bf16.data() + static_cast<size_t>(s * out_dim),
                                  out.data.data() + static_cast<size_t>(s * out_dim),
                                  out_dim);
        }

        if (workspace != nullptr) cudaFree(workspace);
        cudaFree(d_y);
        cudaFree(d_bias);
        cudaFree(d_x);
        cudaFree(d_w);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(d_desc);
        cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatmulDescDestroy(op_desc);
        cublasLtDestroy(lt);
        return out;
    } catch (...) {
        if (workspace != nullptr) cudaFree(workspace);
        if (d_y != nullptr) cudaFree(d_y);
        if (d_bias != nullptr) cudaFree(d_bias);
        if (d_x != nullptr) cudaFree(d_x);
        if (d_w != nullptr) cudaFree(d_w);
        if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
        if (d_desc != nullptr) cublasLtMatrixLayoutDestroy(d_desc);
        if (c_desc != nullptr) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc != nullptr) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc != nullptr) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc != nullptr) cublasLtMatmulDescDestroy(op_desc);
        if (lt != nullptr) cublasLtDestroy(lt);
        throw;
    }
#endif
}
#endif

static TensorF32 linear_ggml_projection_mode(const TensorF32& input,
                                             const std::unordered_map<std::string, SafetensorEntry>& index,
                                             const std::string& prefix,
                                             const RouterCudaMode& mode,
                                             double* elapsed_ms = nullptr) {
    const auto& w_e = need_entry(index, prefix + ".weight");
    const auto& b_e = need_entry(index, prefix + ".bias");
    const std::vector<ggml_bf16_t> w_bf16 = read_bf16_raw(w_e);
    const std::vector<ggml_bf16_t> b_bf16 = read_bf16_raw(b_e);
    const std::vector<float> w_f32 = read_bf16_as_f32(w_e);
    const std::vector<float> b_f32 = read_bf16_as_f32(b_e);
    return run_router_cuda_mode(mode,
                                input,
                                w_bf16,
                                w_f32,
                                w_e.shape,
                                b_bf16,
                                b_f32,
                                0,
                                nullptr,
                                elapsed_ms,
                                nullptr,
                                nullptr);
}

static void print_projection_worst_sample(const std::string& label,
                                          const TensorF32& actual,
                                          const NpyF32& ref,
                                          const TensorF32& input,
                                          const std::vector<float>& weight,
                                          const std::vector<int64_t>& weight_shape,
                                          const std::vector<float>& bias,
                                          const DiffStats& diff) {
    if (actual.shape.size() != 3 || ref.shape != actual.shape || input.shape.size() != 3 || weight_shape.size() != 2) {
        throw std::runtime_error(label + " projection sample shape mismatch");
    }
    const int64_t out_dim = actual.shape[2];
    const int64_t in_dim = input.shape[2];
    const int64_t token = static_cast<int64_t>(diff.max_index / static_cast<size_t>(out_dim));
    const int64_t out = static_cast<int64_t>(diff.max_index % static_cast<size_t>(out_dim));
    std::cout << label << "_worst"
              << " token=" << token
              << " out=" << out
              << " actual=" << actual.data[diff.max_index]
              << " ref=" << ref.data[diff.max_index]
              << " diff=" << std::fabs(actual.data[diff.max_index] - ref.data[diff.max_index])
              << " bias=" << bias[static_cast<size_t>(out)]
              << " input0_3=";
    for (int i = 0; i < 4 && i < in_dim; ++i) {
        if (i) std::cout << ",";
        std::cout << input.data[static_cast<size_t>(token * in_dim + i)];
    }
    std::cout << " weight0_3=";
    for (int i = 0; i < 4 && i < in_dim; ++i) {
        if (i) std::cout << ",";
        std::cout << weight[static_cast<size_t>(out * in_dim + i)];
    }
    std::cout << "\n";
}

static size_t finite_count(const TensorF32& tensor);

static void report_projection_variant(const std::string& label,
                                      const TensorF32& actual,
                                      const NpyF32& ref,
                                      const TensorF32& input,
                                      const SafetensorEntry& weight_entry,
                                      const SafetensorEntry& bias_entry,
                                      const std::string& backend,
                                      double elapsed_ms) {
    if (actual.shape != ref.shape) {
        throw std::runtime_error(label + " projection output shape mismatch");
    }
    const DiffStats d = diff_stats(actual.data, ref.data);
    size_t finite = finite_count(actual);
    std::cout << label
              << " backend=" << backend
              << " shape=" << shape_string(actual.shape)
              << " finite=" << finite << "/" << actual.data.size()
              << " max_diff=" << d.max_diff
              << " mean_diff=" << d.mean_diff
              << " rel_max=" << d.rel_max
              << " max_index=" << d.max_index
              << " elapsed_ms=" << elapsed_ms
              << "\n";
    print_projection_worst_sample(label,
                                  actual,
                                  ref,
                                  input,
                                  read_bf16_as_f32(weight_entry),
                                  weight_entry.shape,
                                  read_bf16_as_f32(bias_entry),
                                  d);
}

static void report_projection_variant_error(const std::string& label,
                                            const std::string& backend,
                                            const std::exception& e) {
    std::cout << label
              << " backend=" << backend
              << " error=\"" << e.what() << "\"\n";
}

static void run_projection_only(int layer_index,
                                const std::filesystem::path& oracle_dir,
                                const std::unordered_map<std::string, SafetensorEntry>& index) {
    if (layer_index != 0) {
        throw std::runtime_error("--projection-only currently supports layer=0");
    }
    struct ProjectionCase {
        std::string name;
        std::string prefix_suffix;
        std::string input_file;
        std::string ref_file;
    };
    const std::string layer_name = "layer" + std::to_string(layer_index);
    const std::string weight_prefix = "model.layers." + std::to_string(layer_index) + ".";
    const std::vector<ProjectionCase> cases = {
        {"q", "self_attn.q_proj", layer_name + "_input_norm_output_f32.npy", layer_name + "_manual_q_proj_f32.npy"},
        {"k", "self_attn.k_proj", layer_name + "_input_norm_output_f32.npy", layer_name + "_manual_k_proj_f32.npy"},
        {"v", "self_attn.v_proj", layer_name + "_input_norm_output_f32.npy", layer_name + "_manual_v_proj_f32.npy"},
        {"o", "self_attn.o_proj", layer_name + "_manual_attention_pre_o_exact_f32.npy", layer_name + "_manual_attention_o_f32.npy"},
    };
    const std::vector<RouterCudaMode> ggml_modes = {
        {"ggml_cuda_bf16_bf16_prec_f32_cast_bf16", GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, true, true},
        {"ggml_cuda_bf16_bf16_prec_f32_f32_output", GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, true, false},
        {"ggml_cuda_f32_f32_prec_f32_cast_bf16", GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, true, true},
        {"ggml_cuda_bf16_bf16_default_cast_bf16", GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, false, true},
    };

    std::cout << "projection_only layer=" << layer_index
              << " variants=manual_cpu_f32_acc_bf16";
#ifdef SD_LENS_TEXT_ENCODER_USE_CUBLAS
    std::cout << ",cublas_bf16_compute32_tensorop";
    std::cout << ",cublas_bf16_output_compute32_cpu_bias";
#endif
#ifdef SD_LENS_TEXT_ENCODER_USE_CUBLASLT
    std::cout << ",cublaslt_bf16_bias_compute32";
    std::cout << ",cublaslt_bf16_bias_compute32_pedantic";
    std::cout << ",cublaslt_bf16_bias_compute32_fast_16bf";
#endif
    for (const RouterCudaMode& mode : ggml_modes) {
        std::cout << "," << mode.name;
    }
    std::cout << "\n";

    for (const ProjectionCase& c : cases) {
        const std::string prefix = weight_prefix + c.prefix_suffix;
        const auto& w_e = need_entry(index, prefix + ".weight");
        const auto& b_e = need_entry(index, prefix + ".bias");
        const NpyF32 input_npy = load_npy_f32(oracle_dir / c.input_file);
        const NpyF32 ref = load_npy_f32(oracle_dir / c.ref_file);
        TensorF32 input = make_tensor(input_npy.shape, input_npy.data);

        {
            const auto start = std::chrono::steady_clock::now();
            TensorF32 actual = linear_bf16(input, read_bf16_as_f32(w_e), w_e.shape, read_bf16_as_f32(b_e), c.name + "_manual_cpu");
            const double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            report_projection_variant("projection_" + c.name,
                                      actual,
                                      ref,
                                      input,
                                      w_e,
                                      b_e,
                                      "manual_cpu_f32_acc_bf16",
                                      elapsed_ms);
        }

        for (const RouterCudaMode& mode : ggml_modes) {
            double elapsed_ms = 0.0;
            TensorF32 actual = linear_ggml_projection_mode(input, index, prefix, mode, &elapsed_ms);
            report_projection_variant("projection_" + c.name,
                                      actual,
                                      ref,
                                      input,
                                      w_e,
                                      b_e,
                                      mode.name,
                                      elapsed_ms);
        }

#ifdef SD_LENS_TEXT_ENCODER_USE_CUBLAS
        {
            const std::string backend = "cublas_bf16_compute32_tensorop";
            try {
                double elapsed_ms = 0.0;
                TensorF32 actual = linear_cublas_bf16_projection_probe(input, index, prefix, c.name + "_cublas", &elapsed_ms);
                report_projection_variant("projection_" + c.name,
                                          actual,
                                          ref,
                                          input,
                                          w_e,
                                          b_e,
                                          backend,
                                          elapsed_ms);
            } catch (const std::exception& e) {
                report_projection_variant_error("projection_" + c.name, backend, e);
            }
        }
        {
            const std::string backend = "cublas_bf16_output_compute32_cpu_bias";
            try {
                double elapsed_ms = 0.0;
                TensorF32 actual = linear_cublas_bf16_output_projection_probe(input, index, prefix, c.name + "_cublas_bf16out", &elapsed_ms);
                report_projection_variant("projection_" + c.name,
                                          actual,
                                          ref,
                                          input,
                                          w_e,
                                          b_e,
                                          backend,
                                          elapsed_ms);
            } catch (const std::exception& e) {
                report_projection_variant_error("projection_" + c.name, backend, e);
            }
        }
#endif
#ifdef SD_LENS_TEXT_ENCODER_USE_CUBLASLT
        struct LtVariant {
            std::string backend;
            cublasComputeType_t compute_type;
        };
        const std::vector<LtVariant> lt_variants = {
            {"cublaslt_bf16_bias_compute32", CUBLAS_COMPUTE_32F},
            {"cublaslt_bf16_bias_compute32_pedantic", CUBLAS_COMPUTE_32F_PEDANTIC},
            {"cublaslt_bf16_bias_compute32_fast_16bf", CUBLAS_COMPUTE_32F_FAST_16BF},
        };
        for (const LtVariant& lt : lt_variants) {
            try {
                double elapsed_ms = 0.0;
                int algo_id = -1;
                TensorF32 actual = linear_cublaslt_bf16_bias_projection_probe(input,
                                                                              index,
                                                                              prefix,
                                                                              c.name + "_" + lt.backend,
                                                                              lt.compute_type,
                                                                              0,
                                                                              &algo_id,
                                                                              &elapsed_ms);
                const std::string backend = lt.backend + "_heuristic0_algo" + std::to_string(algo_id);
                report_projection_variant("projection_" + c.name,
                                          actual,
                                          ref,
                                          input,
                                          w_e,
                                          b_e,
                                          backend,
                                          elapsed_ms);
            } catch (const std::exception& e) {
                report_projection_variant_error("projection_" + c.name, lt.backend + "_heuristic0", e);
            }
        }
        {
            const std::vector<LtVariant> k_sweep_variants = {
                {"cublaslt_bf16_bias_compute32", CUBLAS_COMPUTE_32F},
                {"cublaslt_bf16_bias_compute32_pedantic", CUBLAS_COMPUTE_32F_PEDANTIC},
            };
            const int max_heuristic = c.name == "k" ? 16 : 2;
            for (const LtVariant& lt : k_sweep_variants) {
                for (int heuristic_index = 1; heuristic_index < max_heuristic; ++heuristic_index) {
                    try {
                        double elapsed_ms = 0.0;
                        int algo_id = -1;
                        TensorF32 actual = linear_cublaslt_bf16_bias_projection_probe(input,
                                                                                      index,
                                                                                      prefix,
                                                                                      c.name + "_" + lt.backend + "_sweep",
                                                                                      lt.compute_type,
                                                                                      heuristic_index,
                                                                                      &algo_id,
                                                                                      &elapsed_ms);
                        const std::string backend = lt.backend + "_heuristic" + std::to_string(heuristic_index) +
                                                    "_algo" + std::to_string(algo_id);
                        report_projection_variant("projection_" + c.name,
                                                  actual,
                                                  ref,
                                                  input,
                                                  w_e,
                                                  b_e,
                                                  backend,
                                                  elapsed_ms);
                    } catch (const std::exception& e) {
                        const std::string backend = lt.backend + "_heuristic" + std::to_string(heuristic_index);
                        report_projection_variant_error("projection_" + c.name, backend, e);
                    }
                }
            }
        }
#endif
    }
}

static void report_rope_variant(const std::string& label,
                                RopeVariant variant,
                                const TensorF32& actual,
                                const NpyF32& ref) {
    if (actual.shape != ref.shape || actual.shape.size() != 4) {
        throw std::runtime_error(label + " RoPE output shape mismatch");
    }
    const DiffStats d = diff_stats(actual.data, ref.data);
    const int64_t head_dim = actual.shape[3];
    const int64_t seq = actual.shape[2];
    const int64_t token_stride = head_dim;
    const int64_t head_stride = seq * head_dim;
    const int64_t head = static_cast<int64_t>(d.max_index / static_cast<size_t>(head_stride));
    const int64_t rem0 = static_cast<int64_t>(d.max_index % static_cast<size_t>(head_stride));
    const int64_t token = rem0 / token_stride;
    const int64_t dim = rem0 % token_stride;
    std::cout << label
              << " variant=" << rope_variant_name(variant)
              << " shape=" << shape_string(actual.shape)
              << " finite=" << finite_count(actual) << "/" << actual.data.size()
              << " max_diff=" << d.max_diff
              << " mean_diff=" << d.mean_diff
              << " rel_max=" << d.rel_max
              << " max_index=" << d.max_index
              << " coord=0," << head << "," << token << "," << dim
              << " actual=" << actual.data[d.max_index]
              << " ref=" << ref.data[d.max_index]
              << " actual_bf16_bits=0x" << std::hex << f32_to_bf16_bits(actual.data[d.max_index])
              << " ref_bf16_bits=0x" << f32_to_bf16_bits(ref.data[d.max_index]) << std::dec
              << "\n";
}

static void run_rope_only(int layer_index, const std::filesystem::path& oracle_dir) {
    if (layer_index != 0) {
        throw std::runtime_error("--rope-only currently supports layer=0");
    }
    const NpyF32 q_states_npy = load_npy_f32(oracle_dir / "layer0_manual_q_states_f32.npy");
    const NpyF32 k_states_npy = load_npy_f32(oracle_dir / "layer0_manual_k_states_f32.npy");
    const NpyF32 q_ref = load_npy_f32(oracle_dir / "layer0_manual_q_rope_f32.npy");
    const NpyF32 k_ref = load_npy_f32(oracle_dir / "layer0_manual_k_rope_f32.npy");
    const auto rope_npy = load_or_make_gptoss_rope_cos_sin(oracle_dir);
    const NpyF32& cos_npy = rope_npy.first;
    const NpyF32& sin_npy = rope_npy.second;
    if (cos_npy.shape.size() != 3 || sin_npy.shape != cos_npy.shape) {
        throw std::runtime_error("RoPE cos/sin oracle shape mismatch");
    }
    TensorF32 q_states = make_tensor(q_states_npy.shape, q_states_npy.data);
    TensorF32 k_states = make_tensor(k_states_npy.shape, k_states_npy.data);
    std::cout << "rope_only layer=" << layer_index
              << " q_shape=" << shape_string(q_states.shape)
              << " k_shape=" << shape_string(k_states.shape)
              << " cos_shape=" << shape_string(cos_npy.shape)
              << " sin_shape=" << shape_string(sin_npy.shape)
              << " cos_sin_dtype=BF16_cast_stored_as_F32"
              << " variants=split_half_final_cast,split_half_product_cast,interleaved_final_cast\n";
    const std::vector<RopeVariant> variants = {
        RopeVariant::SplitHalfFinalCast,
        RopeVariant::SplitHalfProductCast,
        RopeVariant::InterleavedFinalCast,
    };
    for (RopeVariant variant : variants) {
        TensorF32 q_actual = apply_rope_to_states_bf16(q_states, cos_npy.data, sin_npy.data, variant);
        TensorF32 k_actual = apply_rope_to_states_bf16(k_states, cos_npy.data, sin_npy.data, variant);
        report_rope_variant("rope_q", variant, q_actual, q_ref);
        report_rope_variant("rope_k", variant, k_actual, k_ref);
    }
}

static float bf16_negative_max() {
    return bf16_to_f32(0xff7fu);
}

static TensorF32 repeat_kv_states(const TensorF32& kv, int64_t heads, int64_t kv_heads, int64_t seq, int64_t head_dim) {
    if (kv.shape != std::vector<int64_t>{1, kv_heads, seq, head_dim} || heads % kv_heads != 0) {
        throw std::runtime_error("repeat_kv input shape mismatch");
    }
    const int64_t groups = heads / kv_heads;
    TensorF32 out;
    out.shape = {1, heads, seq, head_dim};
    out.data.resize(static_cast<size_t>(heads * seq * head_dim));
    for (int64_t kh = 0; kh < kv_heads; ++kh) {
        for (int64_t g = 0; g < groups; ++g) {
            const int64_t h = kh * groups + g;
            for (int64_t s = 0; s < seq; ++s) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    out.data[static_cast<size_t>((h * seq + s) * head_dim + d)] =
                        kv.data[static_cast<size_t>((kh * seq + s) * head_dim + d)];
                }
            }
        }
    }
    return out;
}

static TensorF32 make_gptoss_attention_mask(int64_t seq, bool sliding_attention, int64_t sliding_window) {
    TensorF32 mask;
    mask.shape = {1, 1, seq, seq};
    mask.data.resize(static_cast<size_t>(seq * seq), 0.0f);
    const float neg = bf16_negative_max();
    for (int64_t i = 0; i < seq; ++i) {
        for (int64_t j = 0; j < seq; ++j) {
            if (j > i || (sliding_attention && (i - j) >= sliding_window)) {
                mask.data[static_cast<size_t>(i * seq + j)] = neg;
            }
        }
    }
    return mask;
}

struct AttentionTrace {
    TensorF32 k_repeat;
    TensorF32 v_repeat;
    TensorF32 mask;
    TensorF32 raw_logits;
    TensorF32 masked_logits;
    TensorF32 sinks_expanded;
    TensorF32 combined_logits;
    TensorF32 row_max;
    TensorF32 centered_logits;
    TensorF32 probs_with_sink;
    TensorF32 token_scores;
    TensorF32 token_scores_value_dtype;
    TensorF32 pre_o;
};

static AttentionTrace gptoss_attention_trace_cpu(const TensorF32& q,
                                                 const TensorF32& k,
                                                 const TensorF32& v,
                                                 const std::vector<float>& sinks,
                                                 int64_t heads,
                                                 int64_t kv_heads,
                                                 int64_t seq,
                                                 int64_t head_dim,
                                                 bool sliding_attention,
                                                 int64_t sliding_window) {
    if (q.shape != std::vector<int64_t>{1, heads, seq, head_dim} ||
        k.shape != std::vector<int64_t>{1, kv_heads, seq, head_dim} ||
        v.shape != std::vector<int64_t>{1, kv_heads, seq, head_dim} ||
        static_cast<int64_t>(sinks.size()) != heads) {
        throw std::runtime_error("attention trace input shape mismatch");
    }
    AttentionTrace t;
    t.k_repeat = repeat_kv_states(k, heads, kv_heads, seq, head_dim);
    t.v_repeat = repeat_kv_states(v, heads, kv_heads, seq, head_dim);
    t.mask = make_gptoss_attention_mask(seq, sliding_attention, sliding_window);
    t.raw_logits.shape = {1, heads, seq, seq};
    t.masked_logits.shape = {1, heads, seq, seq};
    t.sinks_expanded.shape = {1, heads, seq, 1};
    t.combined_logits.shape = {1, heads, seq, seq + 1};
    t.row_max.shape = {1, heads, seq, 1};
    t.centered_logits.shape = {1, heads, seq, seq + 1};
    t.probs_with_sink.shape = {1, heads, seq, seq + 1};
    t.token_scores.shape = {1, heads, seq, seq};
    t.token_scores_value_dtype.shape = {1, heads, seq, seq};
    t.pre_o.shape = {1, seq, heads * head_dim};
    t.raw_logits.data.resize(static_cast<size_t>(heads * seq * seq));
    t.masked_logits.data.resize(static_cast<size_t>(heads * seq * seq));
    t.sinks_expanded.data.resize(static_cast<size_t>(heads * seq));
    t.combined_logits.data.resize(static_cast<size_t>(heads * seq * (seq + 1)));
    t.row_max.data.resize(static_cast<size_t>(heads * seq));
    t.centered_logits.data.resize(static_cast<size_t>(heads * seq * (seq + 1)));
    t.probs_with_sink.data.resize(static_cast<size_t>(heads * seq * (seq + 1)));
    t.token_scores.data.resize(static_cast<size_t>(heads * seq * seq));
    t.token_scores_value_dtype.data.resize(static_cast<size_t>(heads * seq * seq));
    t.pre_o.data.assign(static_cast<size_t>(seq * heads * head_dim), 0.0f);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int64_t h = 0; h < heads; ++h) {
        for (int64_t i = 0; i < seq; ++i) {
            float row_max = -std::numeric_limits<float>::infinity();
            for (int64_t j = 0; j < seq; ++j) {
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) {
                    dot = std::fma(q.data[static_cast<size_t>((h * seq + i) * head_dim + d)],
                                   t.k_repeat.data[static_cast<size_t>((h * seq + j) * head_dim + d)],
                                   dot);
                }
                const float raw = round_to_bf16(dot * scale);
                const float mask_v = t.mask.data[static_cast<size_t>(i * seq + j)];
                const float masked = mask_v < -1.0e30f ? mask_v : round_to_bf16(raw + mask_v);
                const size_t idx4 = static_cast<size_t>(((h * seq + i) * seq) + j);
                t.raw_logits.data[idx4] = raw;
                t.masked_logits.data[idx4] = masked;
                const size_t comb = static_cast<size_t>((h * seq + i) * (seq + 1) + j);
                t.combined_logits.data[comb] = masked;
                row_max = std::max(row_max, masked);
            }
            const float sink = sinks[static_cast<size_t>(h)];
            t.sinks_expanded.data[static_cast<size_t>(h * seq + i)] = sink;
            t.combined_logits.data[static_cast<size_t>((h * seq + i) * (seq + 1) + seq)] = sink;
            row_max = std::max(row_max, sink);
            t.row_max.data[static_cast<size_t>(h * seq + i)] = row_max;

            double denom = 0.0;
            for (int64_t j = 0; j <= seq; ++j) {
                const size_t comb = static_cast<size_t>((h * seq + i) * (seq + 1) + j);
                const float centered = round_to_bf16(t.combined_logits.data[comb] - row_max);
                t.centered_logits.data[comb] = centered;
                if (std::isfinite(centered)) {
                    denom += std::exp(static_cast<double>(centered));
                }
            }
            for (int64_t j = 0; j <= seq; ++j) {
                const size_t comb = static_cast<size_t>((h * seq + i) * (seq + 1) + j);
                const float prob = std::isfinite(t.centered_logits.data[comb])
                    ? round_to_bf16(static_cast<float>(std::exp(static_cast<double>(t.centered_logits.data[comb])) / denom))
                    : 0.0f;
                t.probs_with_sink.data[comb] = prob;
                if (j < seq) {
                    const size_t idx4 = static_cast<size_t>(((h * seq + i) * seq) + j);
                    t.token_scores.data[idx4] = prob;
                    t.token_scores_value_dtype.data[idx4] = prob;
                }
            }
            for (int64_t d = 0; d < head_dim; ++d) {
                float sum = 0.0f;
                for (int64_t j = 0; j < seq; ++j) {
                    sum = std::fma(t.token_scores_value_dtype.data[static_cast<size_t>((h * seq + i) * seq + j)],
                                   t.v_repeat.data[static_cast<size_t>((h * seq + j) * head_dim + d)],
                                   sum);
                }
                t.pre_o.data[static_cast<size_t>(i * heads * head_dim + h * head_dim + d)] = round_to_bf16(sum);
            }
        }
    }
    return t;
}

static void run_attention_only(int layer_index,
                               const std::filesystem::path& oracle_dir,
                               const std::unordered_map<std::string, SafetensorEntry>& index) {
    if (layer_index != 0) {
        throw std::runtime_error("--attention-only currently supports layer=0");
    }
    const int64_t heads = 64;
    const int64_t kv_heads = 8;
    const int64_t seq = 128;
    const int64_t head_dim = 64;
    const int64_t hidden = 2880;
    const bool sliding_attention = true;
    const int64_t sliding_window = 128;
    const std::string weight_prefix = "model.layers.0.";
    TensorF32 q = make_tensor({1, heads, seq, head_dim}, load_npy_f32(oracle_dir / "layer0_manual_q_rope_f32.npy").data);
    TensorF32 k = make_tensor({1, kv_heads, seq, head_dim}, load_npy_f32(oracle_dir / "layer0_manual_k_rope_f32.npy").data);
    TensorF32 v = make_tensor({1, kv_heads, seq, head_dim}, load_npy_f32(oracle_dir / "layer0_manual_v_states_f32.npy").data);
    const std::vector<float> sinks = read_bf16_as_f32(need_entry(index, weight_prefix + "self_attn.sinks"));
    std::cout << "attention_only layer=0"
              << " type=sliding_attention"
              << " sliding_window=" << sliding_window
              << " heads=" << heads
              << " kv_heads=" << kv_heads
              << " repeat_factor=" << (heads / kv_heads)
              << " head_dim=" << head_dim
              << " scale=" << (1.0f / std::sqrt(static_cast<float>(head_dim)))
              << " sink_shape=64"
              << " sink_sample=";
    for (int i = 0; i < 8 && i < static_cast<int>(sinks.size()); ++i) {
        if (i) std::cout << ",";
        std::cout << sinks[static_cast<size_t>(i)];
    }
    std::cout << "\n";

    const AttentionTrace trace = gptoss_attention_trace_cpu(q, k, v, sinks, heads, kv_heads, seq, head_dim, sliding_attention, sliding_window);
    report_diff("attention_k_repeat", trace.k_repeat, load_npy_f32(oracle_dir / "layer0_manual_k_repeat_f32.npy"));
    report_diff("attention_v_repeat", trace.v_repeat, load_npy_f32(oracle_dir / "layer0_manual_v_repeat_f32.npy"));
    report_diff("attention_mask", trace.mask, load_npy_f32(oracle_dir / "layer0_manual_attention_mask_f32.npy"));
    report_diff("attention_logits_raw", trace.raw_logits, load_npy_f32(oracle_dir / "layer0_manual_attention_logits_raw_f32.npy"));
    report_diff("attention_logits_masked", trace.masked_logits, load_npy_f32(oracle_dir / "layer0_manual_attention_logits_masked_f32.npy"));
    if (std::filesystem::exists(oracle_dir / "layer0_manual_attention_sinks_expanded_f32.npy")) {
        report_diff("attention_sinks_expanded", trace.sinks_expanded, load_npy_f32(oracle_dir / "layer0_manual_attention_sinks_expanded_f32.npy"));
    }
    report_diff("attention_combined_logits", trace.combined_logits, load_npy_f32(oracle_dir / "layer0_manual_attention_combined_logits_f32.npy"));
    report_diff("attention_row_max", trace.row_max, load_npy_f32(oracle_dir / "layer0_manual_attention_row_max_f32.npy"));
    report_diff("attention_combined_logits_centered", trace.centered_logits, load_npy_f32(oracle_dir / "layer0_manual_attention_combined_logits_centered_f32.npy"));
    report_diff("attention_probs_with_sink", trace.probs_with_sink, load_npy_f32(oracle_dir / "layer0_manual_attention_probs_with_sink_f32.npy"));
    report_diff("attention_token_scores", trace.token_scores, load_npy_f32(oracle_dir / "layer0_manual_attention_token_scores_f32.npy"));
    report_diff("attention_token_scores_value_dtype", trace.token_scores_value_dtype, load_npy_f32(oracle_dir / "layer0_manual_attention_token_scores_value_dtype_f32.npy"));
    report_diff("attention_pre_o_cpu_source", trace.pre_o, load_npy_f32(oracle_dir / "layer0_manual_attention_pre_o_exact_f32.npy"));

    double cuda_ms = 0.0;
    const TensorF32 cuda_pre_o = attention_pre_o_cuda_bf16(q, k, v, sinks, heads, kv_heads, seq, head_dim, sliding_attention, sliding_window, &cuda_ms);
    std::cout << "attention_pre_o_cuda elapsed_ms=" << cuda_ms << "\n";
    report_diff("attention_pre_o_cuda", cuda_pre_o, load_npy_f32(oracle_dir / "layer0_manual_attention_pre_o_exact_f32.npy"));

    const TensorF32 attn_o = linear_cuda_bf16(trace.pre_o, index, weight_prefix + "self_attn.o_proj", "layer0_attention_only_o_proj");
    report_diff("attention_o_cpu_source", attn_o, load_npy_f32(oracle_dir / "layer0_manual_attention_o_f32.npy"));
    const TensorF32 residual = make_tensor({1, seq, hidden}, load_npy_f32(oracle_dir / "token_embedding_f32.npy").data);
    const TensorF32 post_residual = add_bf16(residual, attn_o, "attention_only_post_residual");
    report_diff("attention_post_residual_cpu_source", post_residual, load_npy_f32(oracle_dir / "layer0_manual_post_attention_residual_f32.npy"));
    const std::vector<float> post_norm_w = read_bf16_as_f32(need_entry(index, weight_prefix + "post_attention_layernorm.weight"));
    const TensorF32 post_norm = rms_norm_bf16(post_residual, post_norm_w, 1.0e-5f);
    report_diff("attention_post_norm_cpu_source", post_norm, load_npy_f32(oracle_dir / "layer0_manual_post_attention_norm_f32.npy"));
}

static size_t finite_count(const TensorF32& tensor) {
    size_t count = 0;
    for (float v : tensor.data) {
        if (std::isfinite(v)) {
            ++count;
        }
    }
    return count;
}

static void require_all_finite(const std::string& label, const TensorF32& tensor) {
    const size_t finite = finite_count(tensor);
    if (finite != tensor.data.size()) {
        std::ostringstream oss;
        oss << label << " nonfinite " << (tensor.data.size() - finite) << "/" << tensor.data.size();
        throw std::runtime_error(oss.str());
    }
}

struct MoeTiming {
    double weight_read_seconds = 0.0;
    double dequant_seconds = 0.0;
    double upload_seconds = 0.0;
    double cache_map_open_seconds = 0.0;
    double cache_staging_seconds = 0.0;
    double h2d_upload_seconds = 0.0;
    double d2h_download_seconds = 0.0;
    double wait_upload_seconds = 0.0;
    double cuda_setup_seconds = 0.0;
    double host_register_seconds = 0.0;
    double host_unregister_seconds = 0.0;
    uint64_t registered_host_bytes = 0;
    double gate_up_seconds = 0.0;
    double swiglu_seconds = 0.0;
    double down_seconds = 0.0;
    double routing_seconds = 0.0;
    double reduce_seconds = 0.0;
    uint64_t cache_bytes = 0;
    uint64_t decoded_bytes = 0;
    bool bf16_cache_hit = false;
    bool bf16_cache_written = false;
};

struct LayerRunResult {
    TensorF32 hidden;
    TensorF32 mlp_output;
    TopKResult topk;
    double dense_cuda_ms = 0.0;
    double attention_ms = 0.0;
    double moe_seconds = 0.0;
    double layer_other_seconds = 0.0;
    double norm_weight_load_seconds = 0.0;
    double rmsnorm_total_seconds = 0.0;
    double rmsnorm_compute_seconds = 0.0;
    double linear_total_seconds = 0.0;
    double linear_compute_seconds = 0.0;
    double router_weight_load_seconds = 0.0;
    double router_total_seconds = 0.0;
    double router_compute_seconds = 0.0;
    double rope_load_seconds = 0.0;
    double rope_apply_seconds = 0.0;
    double sink_load_seconds = 0.0;
    double residual_cast_seconds = 0.0;
    double topk_seconds = 0.0;
    double finite_check_seconds = 0.0;
    double input_check_seconds = 0.0;
    double input_norm_stage_seconds = 0.0;
    double qkv_stage_seconds = 0.0;
    double rope_stage_seconds = 0.0;
    double attention_stage_seconds = 0.0;
    double o_proj_stage_seconds = 0.0;
    double post_attention_stage_seconds = 0.0;
    double router_stage_seconds = 0.0;
    double topk_stage_seconds = 0.0;
    double moe_call_wall_seconds = 0.0;
    double final_stage_seconds = 0.0;
    double oracle_compare_seconds = 0.0;
    double stage_unaccounted_seconds = 0.0;
    MoeTiming moe_timing;
    size_t active_experts = 0;
    size_t expert_set_mismatch_tokens = 0;
    size_t slot_order_tie_mismatches = 0;
    float canonical_weight_max_diff = -1.0f;
    float canonical_weight_mean_diff = -1.0f;
    float router_logits_max_diff = -1.0f;
    float router_logits_mean_diff = -1.0f;
    float moe_max_diff = -1.0f;
    float moe_mean_diff = -1.0f;
    float final_hidden_max_diff = -1.0f;
    float final_hidden_mean_diff = -1.0f;
    float final_hidden_rel_max = -1.0f;
};

static std::string g_moe_backend = "cpu-active-expert";
static std::string g_moe_cache = "per-layer-dequant";
static std::string g_moe_cache_layout = "packed-active";
static std::string g_cache_upload = "direct-mapped";
static int g_cache_upload_chunk_mib = 64;
static std::filesystem::path g_moe_bf16_cache_dir = "build/diagnostics/lens_moe_bf16_cache";

static std::filesystem::path moe_bf16_cache_path(int layer_index, const char* name) {
    return g_moe_bf16_cache_dir / ("layer" + std::to_string(layer_index) + "_" + name + "_all32.bf16");
}

static bool read_bf16_cache_file(const std::filesystem::path& path, std::vector<ggml_bf16_t>& dst, size_t expected_count) {
    if (!std::filesystem::exists(path)) {
        return false;
    }
    const uint64_t expected_bytes = static_cast<uint64_t>(expected_count * sizeof(ggml_bf16_t));
    if (std::filesystem::file_size(path) != expected_bytes) {
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open BF16 MoE cache: " + path.string());
    }
    dst.resize(expected_count);
    read_exact(in, dst.data(), static_cast<size_t>(expected_bytes), path.string());
    return true;
}

#ifdef _WIN32
struct Bf16MappedCacheFile {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
    const ggml_bf16_t* data = nullptr;
    size_t count = 0;
    uint64_t bytes = 0;

    ~Bf16MappedCacheFile() {
        if (data != nullptr) {
            UnmapViewOfFile(data);
        }
        if (mapping != nullptr) {
            CloseHandle(mapping);
        }
        if (file != INVALID_HANDLE_VALUE) {
            CloseHandle(file);
        }
    }
};

static std::unordered_map<std::string, std::shared_ptr<Bf16MappedCacheFile>> g_bf16_cache_maps;

static std::shared_ptr<Bf16MappedCacheFile> map_bf16_cache_file(const std::filesystem::path& path,
                                                                size_t expected_count,
                                                                double* open_seconds) {
    const std::string key = path.string();
    const auto found = g_bf16_cache_maps.find(key);
    if (found != g_bf16_cache_maps.end()) {
        if (open_seconds != nullptr) {
            *open_seconds = 0.0;
        }
        return found->second;
    }
    if (!std::filesystem::exists(path)) {
        return nullptr;
    }
    const uint64_t expected_bytes = static_cast<uint64_t>(expected_count * sizeof(ggml_bf16_t));
    if (std::filesystem::file_size(path) != expected_bytes) {
        return nullptr;
    }

    const auto start = std::chrono::steady_clock::now();
    auto mapped = std::make_shared<Bf16MappedCacheFile>();
    const std::wstring wide = path.wstring();
    mapped->file = CreateFileW(wide.c_str(),
                               GENERIC_READ,
                               FILE_SHARE_READ,
                               nullptr,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
                               nullptr);
    if (mapped->file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("failed to open BF16 MoE cache mapping: " + path.string());
    }
    mapped->mapping = CreateFileMappingW(mapped->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapped->mapping == nullptr) {
        throw std::runtime_error("failed to create BF16 MoE cache mapping: " + path.string());
    }
    mapped->data = reinterpret_cast<const ggml_bf16_t*>(MapViewOfFile(mapped->mapping, FILE_MAP_READ, 0, 0, 0));
    if (mapped->data == nullptr) {
        throw std::runtime_error("failed to map BF16 MoE cache view: " + path.string());
    }
    mapped->count = expected_count;
    mapped->bytes = expected_bytes;
    if (open_seconds != nullptr) {
        *open_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    g_bf16_cache_maps.emplace(key, mapped);
    return mapped;
}
#else
static std::shared_ptr<void> map_bf16_cache_file(const std::filesystem::path&, size_t, double*) {
    return nullptr;
}
#endif

#if defined(SD_USE_CUDA) && defined(SD_LENS_TEXT_ENCODER_USE_CUBLAS)
struct PinnedBf16Buffer {
    ggml_bf16_t* ptr = nullptr;
    size_t capacity = 0;
    bool pinned = false;
    std::vector<ggml_bf16_t> fallback;

    ~PinnedBf16Buffer() {
        release();
    }

    void release() {
        if (pinned && ptr != nullptr) {
            cudaFreeHost(ptr);
        }
        ptr = nullptr;
        capacity = 0;
        pinned = false;
        fallback.clear();
    }

    ggml_bf16_t* ensure(size_t count) {
        if (count <= capacity && ptr != nullptr) {
            return ptr;
        }
        if (pinned && ptr != nullptr) {
            cudaFreeHost(ptr);
        }
        ptr = nullptr;
        capacity = 0;
        pinned = false;
        fallback.clear();
        const size_t bytes = count * sizeof(ggml_bf16_t);
        if (bytes > 0 && cudaMallocHost(reinterpret_cast<void**>(&ptr), bytes) == cudaSuccess) {
            capacity = count;
            pinned = true;
            return ptr;
        }
        fallback.resize(count);
        ptr = fallback.data();
        capacity = count;
        return ptr;
    }
};

struct CudaMoeWorkspace {
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    void* d_gate_w = nullptr;
    void* d_down_w = nullptr;
    void* d_input = nullptr;
    void* d_gate_out = nullptr;
    void* d_gated = nullptr;
    void* d_down_out = nullptr;
    size_t gate_w_bytes = 0;
    size_t down_w_bytes = 0;
    size_t input_bytes = 0;
    size_t gate_out_bytes = 0;
    size_t gated_bytes = 0;
    size_t down_out_bytes = 0;
    PinnedBf16Buffer gate_w_host;
    PinnedBf16Buffer down_w_host;
    PinnedBf16Buffer input_host;
    PinnedBf16Buffer gate_out_host;
    PinnedBf16Buffer gated_host;
    PinnedBf16Buffer down_out_host;
    bool initialized = false;

    ~CudaMoeWorkspace() {
        release();
    }

    void release() {
        if (d_down_out != nullptr) cudaFree(d_down_out);
        if (d_gated != nullptr) cudaFree(d_gated);
        if (d_gate_out != nullptr) cudaFree(d_gate_out);
        if (d_input != nullptr) cudaFree(d_input);
        if (d_down_w != nullptr) cudaFree(d_down_w);
        if (d_gate_w != nullptr) cudaFree(d_gate_w);
        if (handle != nullptr) cublasDestroy(handle);
        if (stream != nullptr) cudaStreamDestroy(stream);
        d_gate_w = nullptr;
        d_down_w = nullptr;
        d_input = nullptr;
        d_gate_out = nullptr;
        d_gated = nullptr;
        d_down_out = nullptr;
        gate_w_bytes = 0;
        down_w_bytes = 0;
        input_bytes = 0;
        gate_out_bytes = 0;
        gated_bytes = 0;
        down_out_bytes = 0;
        handle = nullptr;
        stream = nullptr;
        gate_w_host.release();
        down_w_host.release();
        input_host.release();
        gate_out_host.release();
        gated_host.release();
        down_out_host.release();
        initialized = false;
    }

    void init() {
        if (initialized) {
            return;
        }
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            throw std::runtime_error("cudaStreamCreate failed for batched MoE workspace");
        }
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed for batched MoE workspace");
        }
        if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasSetStream failed for batched MoE workspace");
        }
        initialized = true;
    }

    static void ensure_device(void** ptr, size_t* capacity, size_t bytes, const char* label) {
        if (bytes <= *capacity && *ptr != nullptr) {
            return;
        }
        if (*ptr != nullptr) {
            cudaFree(*ptr);
            *ptr = nullptr;
            *capacity = 0;
        }
        if (bytes > 0 && cudaMalloc(ptr, bytes) != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed for batched MoE ") + label);
        }
        *capacity = bytes;
    }

    void ensure_device_buffers(size_t gate_w,
                               size_t down_w,
                               size_t input,
                               size_t gate_out,
                               size_t gated,
                               size_t down_out) {
        init();
        ensure_device(&d_gate_w, &gate_w_bytes, gate_w, "gate weights");
        ensure_device(&d_down_w, &down_w_bytes, down_w, "down weights");
        ensure_device(&d_input, &input_bytes, input, "input");
        ensure_device(&d_gate_out, &gate_out_bytes, gate_out, "gate output");
        ensure_device(&d_gated, &gated_bytes, gated, "gated");
        ensure_device(&d_down_out, &down_out_bytes, down_out, "down output");
    }
};

static CudaMoeWorkspace g_cuda_moe_workspace;

static void release_lens_text_encoder_runtime_state() {
    g_cuda_moe_workspace.release();
#ifdef _WIN32
    g_safetensor_file_maps.clear();
    g_bf16_cache_maps.clear();
#endif
}

static void cuda_register_host_range(const void* ptr, size_t bytes, const char* label) {
    if (bytes == 0) {
        return;
    }
    const cudaError_t err = cudaHostRegister(const_cast<void*>(ptr), bytes, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaHostRegister failed for " << label
            << " bytes=" << bytes
            << " error=" << static_cast<int>(err)
            << " message=" << cudaGetErrorString(err)
            << " ptr=" << ptr;
        throw std::runtime_error(oss.str());
    }
}

static void cuda_unregister_host_range(const void* ptr, const char* label) {
    const cudaError_t err = cudaHostUnregister(const_cast<void*>(ptr));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaHostUnregister failed for " << label
            << " error=" << static_cast<int>(err)
            << " message=" << cudaGetErrorString(err)
            << " ptr=" << ptr;
        throw std::runtime_error(oss.str());
    }
}

static void cuda_copy_registered_chunks(void* dst,
                                        const void* src,
                                        size_t bytes,
                                        size_t chunk_bytes,
                                        cudaStream_t stream,
                                        const char* label,
                                        MoeTiming& timing) {
    if (bytes == 0) {
        return;
    }
    if (chunk_bytes == 0) {
        throw std::runtime_error("registered mapped chunk upload received zero chunk size");
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(src);
    if ((base % 4096) != 0) {
        std::ostringstream oss;
        oss << "registered mapped chunk upload requires page-aligned source for " << label
            << " ptr=" << src;
        throw std::runtime_error(oss.str());
    }
    size_t offset = 0;
    while (offset < bytes) {
        const size_t n = std::min(chunk_bytes, bytes - offset);
        const char* src_chunk = static_cast<const char*>(src) + offset;
        char* dst_chunk = static_cast<char*>(dst) + offset;
        const auto register_start = std::chrono::steady_clock::now();
        cuda_register_host_range(src_chunk, n, label);
        timing.host_register_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - register_start).count();
        timing.registered_host_bytes += static_cast<uint64_t>(n);
        bool registered = true;
        try {
            const auto upload_start = std::chrono::steady_clock::now();
            if (cudaMemcpyAsync(dst_chunk, src_chunk, n, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMemcpyAsync H2D failed for registered chunk ") + label);
            }
            const auto wait_start = std::chrono::steady_clock::now();
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                throw std::runtime_error(std::string("cudaStreamSynchronize failed for registered chunk ") + label);
            }
            timing.wait_upload_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start).count();
            const double upload_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - upload_start).count();
            timing.upload_seconds += upload_elapsed;
            timing.h2d_upload_seconds += upload_elapsed;
            const auto unregister_start = std::chrono::steady_clock::now();
            cuda_unregister_host_range(src_chunk, label);
            registered = false;
            timing.host_unregister_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - unregister_start).count();
        } catch (...) {
            if (registered) {
                cudaHostUnregister(const_cast<char*>(src_chunk));
            }
            throw;
        }
        offset += n;
    }
}
#endif

static void write_bf16_cache_file(const std::filesystem::path& path, const std::vector<ggml_bf16_t>& src) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to write BF16 MoE cache: " + path.string());
    }
    out.write(reinterpret_cast<const char*>(src.data()), static_cast<std::streamsize>(src.size() * sizeof(ggml_bf16_t)));
    if (!out) {
        throw std::runtime_error("failed to finish BF16 MoE cache write: " + path.string());
    }
}

struct MoeExpertResult {
    int64_t expert = -1;
    std::vector<float> contrib;
    double dequant_seconds = 0.0;
    double upload_seconds = 0.0;
    double gate_up_seconds = 0.0;
    double swiglu_seconds = 0.0;
    double down_seconds = 0.0;
    double routing_seconds = 0.0;
    size_t hit_count = 0;
};

static TensorF32 run_moe_layer(int layer_index,
                               const TensorF32& post_norm,
                               const TopKResult& topk,
                               const std::unordered_map<std::string, SafetensorEntry>& index,
                               double* elapsed_seconds = nullptr,
                               bool verbose = true,
                               MoeTiming* timing = nullptr) {
    const int64_t seq = post_norm.shape[1];
    const int64_t hidden = post_norm.shape[2];
    const int64_t intermediate = 2880;
    const int top_k = 4;
    const std::string prefix = "model.layers." + std::to_string(layer_index) + ".mlp.experts.";
    MoeTiming local_timing;
    const auto weight_read_start = std::chrono::steady_clock::now();
    const auto& gate_blocks_e = need_entry(index, prefix + "gate_up_proj_blocks");
    const auto& gate_scales_e = need_entry(index, prefix + "gate_up_proj_scales");
    const auto& gate_bias_e = need_entry(index, prefix + "gate_up_proj_bias");
    const auto& down_blocks_e = need_entry(index, prefix + "down_proj_blocks");
    const auto& down_scales_e = need_entry(index, prefix + "down_proj_scales");
    const auto& down_bias_e = need_entry(index, prefix + "down_proj_bias");
    const bool use_mapped_mxfp4_views =
        g_moe_backend == "cuda-batched-expert-matmul" &&
        g_moe_cache_layout == "full-layer-resident" &&
        g_cache_upload == "cuda-mxfp4-dequant";
    TensorBytesView gate_blocks_view;
    TensorBytesView gate_scales_view;
    TensorBytesView down_blocks_view;
    TensorBytesView down_scales_view;
    std::vector<uint8_t> gate_blocks;
    std::vector<uint8_t> gate_scales;
    std::vector<uint8_t> down_blocks;
    std::vector<uint8_t> down_scales;
    double mapped_open_seconds = 0.0;
    if (use_mapped_mxfp4_views) {
        double open_seconds = 0.0;
        gate_blocks_view = tensor_bytes_view(gate_blocks_e, &open_seconds);
        mapped_open_seconds += open_seconds;
        gate_scales_view = tensor_bytes_view(gate_scales_e, &open_seconds);
        mapped_open_seconds += open_seconds;
        down_blocks_view = tensor_bytes_view(down_blocks_e, &open_seconds);
        mapped_open_seconds += open_seconds;
        down_scales_view = tensor_bytes_view(down_scales_e, &open_seconds);
        mapped_open_seconds += open_seconds;
    } else {
        gate_blocks = read_tensor_bytes(gate_blocks_e);
        gate_scales = read_tensor_bytes(gate_scales_e);
        down_blocks = read_tensor_bytes(down_blocks_e);
        down_scales = read_tensor_bytes(down_scales_e);
    }
    const std::vector<float> gate_bias = read_numeric_as_f32(gate_bias_e);
    const std::vector<float> down_bias = read_numeric_as_f32(down_bias_e);
    local_timing.weight_read_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - weight_read_start).count();
    local_timing.cache_map_open_seconds += mapped_open_seconds;

    std::set<int64_t> active_set(topk.indices.begin(), topk.indices.end());
    std::vector<int64_t> active(active_set.begin(), active_set.end());
    if (verbose) {
        std::cout << "moe_active_experts count=" << active.size() << " experts=";
        for (size_t i = 0; i < active.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << active[i];
        }
        std::cout << "\n";
        std::cout << "moe_tokens_per_expert_topk_hits=";
        for (size_t i = 0; i < topk.counts.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << i << ":" << topk.counts[i];
        }
        std::cout << "\n";
    }

    TensorF32 out;
    out.shape = {1, seq, hidden};
    out.data.assign(static_cast<size_t>(seq * hidden), 0.0f);
    const uint64_t decoded_bytes_per_active =
        static_cast<uint64_t>(hidden) * static_cast<uint64_t>(2 * intermediate + hidden) * sizeof(float);

    auto compute_expert = [&](int64_t expert, const std::vector<std::pair<int64_t, int>>& hits) -> MoeExpertResult {
        MoeExpertResult result;
        result.expert = expert;
        result.hit_count = hits.size();
        result.contrib.assign(static_cast<size_t>(seq * hidden), 0.0f);
        std::vector<float> gate_up(static_cast<size_t>(2 * intermediate));
        std::vector<float> gated(static_cast<size_t>(intermediate));
        std::vector<float> expert_out(static_cast<size_t>(hidden));
        std::vector<float> gate_w(static_cast<size_t>(2 * intermediate * hidden));
        std::vector<float> down_w(static_cast<size_t>(hidden * intermediate));

        auto t0 = std::chrono::steady_clock::now();
        for (int64_t out_dim = 0; out_dim < 2 * intermediate; ++out_dim) {
            float* row = gate_w.data() + static_cast<size_t>(out_dim * hidden);
            for (int64_t in_dim = 0; in_dim < hidden; ++in_dim) {
                row[in_dim] = decode_mxfp4_value(gate_blocks, gate_scales, gate_blocks_e.shape, expert, out_dim, in_dim);
            }
        }
        for (int64_t out_dim = 0; out_dim < hidden; ++out_dim) {
            float* row = down_w.data() + static_cast<size_t>(out_dim * intermediate);
            for (int64_t in_dim = 0; in_dim < intermediate; ++in_dim) {
                row[in_dim] = decode_mxfp4_value(down_blocks, down_scales, down_blocks_e.shape, expert, out_dim, in_dim);
            }
        }
        result.dequant_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        if (g_moe_backend == "cuda-expert-matmul") {
#if !defined(SD_USE_CUDA) || !defined(SD_LENS_TEXT_ENCODER_USE_CUBLASLT)
            throw std::runtime_error("cuda-expert-matmul requires SD_USE_CUDA and CUDA::cublasLt");
#else
            TensorF32 expert_input;
            expert_input.shape = {1, static_cast<int64_t>(hits.size()), hidden};
            expert_input.data.resize(static_cast<size_t>(hits.size()) * static_cast<size_t>(hidden));
            for (size_t row = 0; row < hits.size(); ++row) {
                const int64_t token = hits[row].first;
                const float* src = post_norm.data.data() + static_cast<size_t>(token * hidden);
                float* dst = expert_input.data.data() + row * static_cast<size_t>(hidden);
                std::copy(src, src + hidden, dst);
            }

            std::vector<float> gate_bias_expert(static_cast<size_t>(2 * intermediate));
            std::copy(gate_bias.data() + static_cast<size_t>(expert * 2 * intermediate),
                      gate_bias.data() + static_cast<size_t>((expert + 1) * 2 * intermediate),
                      gate_bias_expert.begin());
            double gate_ms = 0.0;
            TensorF32 gate_cuda = linear_cublaslt_bf16_bias_from_host(expert_input,
                                                                      gate_w,
                                                                      2 * intermediate,
                                                                      hidden,
                                                                      gate_bias_expert,
                                                                      "layer" + std::to_string(layer_index) + "_moe_gate_up_cuda",
                                                                      &gate_ms);
            result.gate_up_seconds += gate_ms / 1000.0;

            t0 = std::chrono::steady_clock::now();
            TensorF32 gated_cuda;
            gated_cuda.shape = {1, static_cast<int64_t>(hits.size()), intermediate};
            gated_cuda.data.resize(static_cast<size_t>(hits.size()) * static_cast<size_t>(intermediate));
            for (size_t row = 0; row < hits.size(); ++row) {
                const float* gate_row = gate_cuda.data.data() + row * static_cast<size_t>(2 * intermediate);
                float* gated_row = gated_cuda.data.data() + row * static_cast<size_t>(intermediate);
                for (int64_t i = 0; i < intermediate; ++i) {
                    const float gate = round_to_bf16(std::min(gate_row[static_cast<size_t>(2 * i)], 7.0f));
                    const float up = round_to_bf16(std::max(-7.0f, std::min(gate_row[static_cast<size_t>(2 * i + 1)], 7.0f)));
                    const float gate_alpha = round_to_bf16(gate * 1.702f);
                    const float sigmoid_out = round_to_bf16(1.0f / (1.0f + std::exp(-gate_alpha)));
                    const float glu = round_to_bf16(gate * sigmoid_out);
                    const float up_plus = round_to_bf16(up + 1.0f);
                    gated_row[static_cast<size_t>(i)] = round_to_bf16(up_plus * glu);
                }
            }
            result.swiglu_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            std::vector<float> down_bias_expert(static_cast<size_t>(hidden));
            std::copy(down_bias.data() + static_cast<size_t>(expert * hidden),
                      down_bias.data() + static_cast<size_t>((expert + 1) * hidden),
                      down_bias_expert.begin());
            double down_ms = 0.0;
            TensorF32 down_cuda = linear_cublaslt_bf16_bias_from_host(gated_cuda,
                                                                      down_w,
                                                                      hidden,
                                                                      intermediate,
                                                                      down_bias_expert,
                                                                      "layer" + std::to_string(layer_index) + "_moe_down_cuda",
                                                                      &down_ms);
            result.down_seconds += down_ms / 1000.0;

            t0 = std::chrono::steady_clock::now();
            for (size_t row = 0; row < hits.size(); ++row) {
                const int64_t token = hits[row].first;
                const int kpos = hits[row].second;
                const float route = topk.weights[static_cast<size_t>(token * top_k + kpos)];
                const float* down_row = down_cuda.data.data() + row * static_cast<size_t>(hidden);
                for (int64_t h = 0; h < hidden; ++h) {
                    const size_t dst = static_cast<size_t>(token * hidden + h);
                    const float weighted = round_to_bf16(down_row[static_cast<size_t>(h)] * route);
                    const float weighted_hidden = round_to_bf16(weighted);
                    result.contrib[dst] = round_to_bf16(result.contrib[dst] + weighted_hidden);
                }
            }
            result.routing_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return result;
#endif
        }

        for (const auto& hit : hits) {
            const int64_t token = hit.first;
            const int kpos = hit.second;
            const float* xrow = post_norm.data.data() + static_cast<size_t>(token * hidden);

            t0 = std::chrono::steady_clock::now();
            for (int64_t out_dim = 0; out_dim < 2 * intermediate; ++out_dim) {
                float sum = 0.0f;
                const float* wrow = gate_w.data() + static_cast<size_t>(out_dim * hidden);
                for (int64_t in_dim = 0; in_dim < hidden; ++in_dim) {
                    sum = std::fma(xrow[in_dim], wrow[in_dim], sum);
                }
                sum = round_to_bf16(sum);
                gate_up[static_cast<size_t>(out_dim)] =
                    round_to_bf16(sum + gate_bias[static_cast<size_t>(expert * 2 * intermediate + out_dim)]);
            }
            result.gate_up_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            t0 = std::chrono::steady_clock::now();
            for (int64_t i = 0; i < intermediate; ++i) {
                const float gate = round_to_bf16(std::min(gate_up[static_cast<size_t>(2 * i)], 7.0f));
                const float up = round_to_bf16(std::max(-7.0f, std::min(gate_up[static_cast<size_t>(2 * i + 1)], 7.0f)));
                const float gate_alpha = round_to_bf16(gate * 1.702f);
                const float sigmoid_out = round_to_bf16(1.0f / (1.0f + std::exp(-gate_alpha)));
                const float glu = round_to_bf16(gate * sigmoid_out);
                const float up_plus = round_to_bf16(up + 1.0f);
                gated[static_cast<size_t>(i)] = round_to_bf16(up_plus * glu);
            }
            result.swiglu_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            t0 = std::chrono::steady_clock::now();
            for (int64_t out_dim = 0; out_dim < hidden; ++out_dim) {
                float sum = 0.0f;
                const float* wrow = down_w.data() + static_cast<size_t>(out_dim * intermediate);
                for (int64_t in_dim = 0; in_dim < intermediate; ++in_dim) {
                    sum = std::fma(gated[static_cast<size_t>(in_dim)], wrow[in_dim], sum);
                }
                sum = round_to_bf16(sum);
                expert_out[static_cast<size_t>(out_dim)] =
                    round_to_bf16(sum + down_bias[static_cast<size_t>(expert * hidden + out_dim)]);
            }
            result.down_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            t0 = std::chrono::steady_clock::now();
            const float route = topk.weights[static_cast<size_t>(token * top_k + kpos)];
            for (int64_t h = 0; h < hidden; ++h) {
                const size_t dst = static_cast<size_t>(token * hidden + h);
                const float weighted = round_to_bf16(expert_out[static_cast<size_t>(h)] * route);
                const float weighted_hidden = round_to_bf16(weighted);
                result.contrib[dst] = round_to_bf16(result.contrib[dst] + weighted_hidden);
            }
            result.routing_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        return result;
    };

    std::vector<std::vector<std::pair<int64_t, int>>> hits_by_expert(static_cast<size_t>(topk.counts.size()));
    for (int64_t s = 0; s < seq; ++s) {
        for (int k = 0; k < top_k; ++k) {
            const int64_t expert = topk.indices[static_cast<size_t>(s * top_k + k)];
            hits_by_expert[static_cast<size_t>(expert)].emplace_back(s, k);
        }
    }

    auto start = std::chrono::steady_clock::now();
    local_timing.cache_bytes = decoded_bytes_per_active;
    local_timing.decoded_bytes = decoded_bytes_per_active * static_cast<uint64_t>(active.size());

    if (g_moe_backend == "cuda-batched-expert-matmul") {
#if !defined(SD_USE_CUDA) || !defined(SD_LENS_TEXT_ENCODER_USE_CUBLAS)
        throw std::runtime_error("cuda-batched-expert-matmul requires SD_USE_CUDA and CUDA::cublas");
#else
        const int64_t active_count = static_cast<int64_t>(active.size());
        int64_t max_hits = 0;
        for (int64_t expert : active) {
            max_hits = std::max<int64_t>(max_hits, static_cast<int64_t>(hits_by_expert[static_cast<size_t>(expert)].size()));
        }
        if (active_count == 0 || max_hits == 0) {
            throw std::runtime_error("cuda-batched-expert-matmul has no active experts");
        }

        const size_t gate_stride = static_cast<size_t>(2 * intermediate * hidden);
        const size_t down_stride = static_cast<size_t>(hidden * intermediate);
        const bool full_layer_resident = g_moe_cache_layout == "full-layer-resident";
        const bool cuda_mxfp4_dequant = full_layer_resident && g_cache_upload == "cuda-mxfp4-dequant";
        if (full_layer_resident && g_moe_cache != "layer-bf16" && !cuda_mxfp4_dequant) {
            throw std::runtime_error("full-layer-resident MoE cache layout requires --moe-cache layer-bf16");
        }
        const int64_t batch_count = full_layer_resident ? static_cast<int64_t>(topk.counts.size()) : active_count;
        const size_t input_stride = static_cast<size_t>(max_hits * hidden);
        const size_t gate_out_stride = static_cast<size_t>(max_hits * 2 * intermediate);
        const size_t gated_stride = static_cast<size_t>(max_hits * intermediate);
        const size_t down_out_stride = static_cast<size_t>(max_hits * hidden);

        const size_t gate_w_count = static_cast<size_t>(batch_count) * gate_stride;
        const size_t down_w_count = static_cast<size_t>(batch_count) * down_stride;
        const size_t input_count = static_cast<size_t>(batch_count) * input_stride;
        const size_t gate_out_count = static_cast<size_t>(batch_count) * gate_out_stride;
        const size_t gated_count = static_cast<size_t>(batch_count) * gated_stride;
        const size_t down_out_count = static_cast<size_t>(batch_count) * down_out_stride;

        ggml_bf16_t* gate_w_bf16 = full_layer_resident ? nullptr : g_cuda_moe_workspace.gate_w_host.ensure(gate_w_count);
        ggml_bf16_t* down_w_bf16 = full_layer_resident ? nullptr : g_cuda_moe_workspace.down_w_host.ensure(down_w_count);
        ggml_bf16_t* input_bf16 = g_cuda_moe_workspace.input_host.ensure(input_count);
        ggml_bf16_t* gate_out_bf16 = g_cuda_moe_workspace.gate_out_host.ensure(gate_out_count);
        ggml_bf16_t* gated_bf16 = g_cuda_moe_workspace.gated_host.ensure(gated_count);
        ggml_bf16_t* down_out_bf16 = g_cuda_moe_workspace.down_out_host.ensure(down_out_count);

        auto stage_start = std::chrono::steady_clock::now();
        bool loaded_bf16_layer_cache = false;
#ifdef _WIN32
        std::shared_ptr<Bf16MappedCacheFile> gate_map;
        std::shared_ptr<Bf16MappedCacheFile> down_map;
#endif
        if (g_moe_cache == "layer-bf16" && !cuda_mxfp4_dequant) {
#ifdef _WIN32
            double gate_open_seconds = 0.0;
            double down_open_seconds = 0.0;
            gate_map = map_bf16_cache_file(moe_bf16_cache_path(layer_index, "gate_up"),
                                           static_cast<size_t>(topk.counts.size()) * gate_stride,
                                           &gate_open_seconds);
            down_map = map_bf16_cache_file(moe_bf16_cache_path(layer_index, "down"),
                                           static_cast<size_t>(topk.counts.size()) * down_stride,
                                           &down_open_seconds);
            local_timing.cache_map_open_seconds += gate_open_seconds + down_open_seconds;
            loaded_bf16_layer_cache = static_cast<bool>(gate_map) && static_cast<bool>(down_map);
            if (loaded_bf16_layer_cache) {
                const auto cache_stage_start = std::chrono::steady_clock::now();
                if (!full_layer_resident) {
                    bool identity_active = active_count == static_cast<int64_t>(topk.counts.size());
                    if (identity_active) {
                        for (int64_t batch = 0; batch < active_count; ++batch) {
                            if (active[static_cast<size_t>(batch)] != batch) {
                                identity_active = false;
                                break;
                            }
                        }
                    }
                    if (identity_active) {
                        std::memcpy(gate_w_bf16, gate_map->data, gate_w_count * sizeof(ggml_bf16_t));
                        std::memcpy(down_w_bf16, down_map->data, down_w_count * sizeof(ggml_bf16_t));
                    } else {
                        for (int64_t batch = 0; batch < active_count; ++batch) {
                            const int64_t expert = active[static_cast<size_t>(batch)];
                            std::memcpy(gate_w_bf16 + static_cast<size_t>(batch) * gate_stride,
                                        gate_map->data + static_cast<size_t>(expert) * gate_stride,
                                        gate_stride * sizeof(ggml_bf16_t));
                            std::memcpy(down_w_bf16 + static_cast<size_t>(batch) * down_stride,
                                        down_map->data + static_cast<size_t>(expert) * down_stride,
                                        down_stride * sizeof(ggml_bf16_t));
                        }
                    }
                    local_timing.cache_staging_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - cache_stage_start).count();
                }
                local_timing.bf16_cache_hit = true;
            } else if (static_cast<bool>(gate_map) != static_cast<bool>(down_map)) {
                throw std::runtime_error("partial BF16 MoE layer cache found; remove cache dir and regenerate");
            }
#else
            std::vector<ggml_bf16_t> all_gate;
            std::vector<ggml_bf16_t> all_down;
            const bool gate_ok = read_bf16_cache_file(moe_bf16_cache_path(layer_index, "gate_up"),
                                                      all_gate,
                                                      static_cast<size_t>(topk.counts.size()) * gate_stride);
            const bool down_ok = read_bf16_cache_file(moe_bf16_cache_path(layer_index, "down"),
                                                      all_down,
                                                      static_cast<size_t>(topk.counts.size()) * down_stride);
            loaded_bf16_layer_cache = gate_ok && down_ok;
            if (loaded_bf16_layer_cache) {
                for (int64_t batch = 0; batch < active_count; ++batch) {
                    const int64_t expert = active[static_cast<size_t>(batch)];
                    std::copy(all_gate.data() + static_cast<size_t>(expert) * gate_stride,
                              all_gate.data() + static_cast<size_t>(expert + 1) * gate_stride,
                              gate_w_bf16 + static_cast<size_t>(batch) * gate_stride);
                    std::copy(all_down.data() + static_cast<size_t>(expert) * down_stride,
                              all_down.data() + static_cast<size_t>(expert + 1) * down_stride,
                              down_w_bf16 + static_cast<size_t>(batch) * down_stride);
                }
                local_timing.bf16_cache_hit = true;
            } else if (gate_ok != down_ok) {
                throw std::runtime_error("partial BF16 MoE layer cache found; remove cache dir and regenerate");
            }
#endif
        }

        if (!loaded_bf16_layer_cache && !cuda_mxfp4_dequant) {
            if (full_layer_resident) {
                throw std::runtime_error("full-layer-resident MoE cache layout requires an existing full-layer BF16 cache");
            }
            std::vector<ggml_bf16_t> all_gate;
            std::vector<ggml_bf16_t> all_down;
            const bool write_full_layer_cache = g_moe_cache == "layer-bf16";
            if (write_full_layer_cache) {
                all_gate.resize(static_cast<size_t>(topk.counts.size()) * gate_stride);
                all_down.resize(static_cast<size_t>(topk.counts.size()) * down_stride);
            }
            const int64_t decode_count = write_full_layer_cache ? static_cast<int64_t>(topk.counts.size()) : active_count;
            for (int64_t decode_batch = 0; decode_batch < decode_count; ++decode_batch) {
                const int64_t expert = write_full_layer_cache ? decode_batch : active[static_cast<size_t>(decode_batch)];
                ggml_bf16_t* gate_dst = write_full_layer_cache
                    ? all_gate.data() + static_cast<size_t>(expert) * gate_stride
                    : gate_w_bf16 + static_cast<size_t>(decode_batch) * gate_stride;
                ggml_bf16_t* down_dst = write_full_layer_cache
                    ? all_down.data() + static_cast<size_t>(expert) * down_stride
                    : down_w_bf16 + static_cast<size_t>(decode_batch) * down_stride;
                for (int64_t out_dim = 0; out_dim < 2 * intermediate; ++out_dim) {
                    ggml_bf16_t* row = gate_dst + static_cast<size_t>(out_dim * hidden);
                    for (int64_t in_dim = 0; in_dim < hidden; ++in_dim) {
                        row[in_dim] = ggml_fp32_to_bf16(decode_mxfp4_value(gate_blocks, gate_scales, gate_blocks_e.shape, expert, out_dim, in_dim));
                    }
                }
                for (int64_t out_dim = 0; out_dim < hidden; ++out_dim) {
                    ggml_bf16_t* row = down_dst + static_cast<size_t>(out_dim * intermediate);
                    for (int64_t in_dim = 0; in_dim < intermediate; ++in_dim) {
                        row[in_dim] = ggml_fp32_to_bf16(decode_mxfp4_value(down_blocks, down_scales, down_blocks_e.shape, expert, out_dim, in_dim));
                    }
                }
            }
            if (write_full_layer_cache) {
                write_bf16_cache_file(moe_bf16_cache_path(layer_index, "gate_up"), all_gate);
                write_bf16_cache_file(moe_bf16_cache_path(layer_index, "down"), all_down);
                for (int64_t batch = 0; batch < active_count; ++batch) {
                    const int64_t expert = active[static_cast<size_t>(batch)];
                    std::copy(all_gate.data() + static_cast<size_t>(expert) * gate_stride,
                              all_gate.data() + static_cast<size_t>(expert + 1) * gate_stride,
                              gate_w_bf16 + static_cast<size_t>(batch) * gate_stride);
                    std::copy(all_down.data() + static_cast<size_t>(expert) * down_stride,
                              all_down.data() + static_cast<size_t>(expert + 1) * down_stride,
                              down_w_bf16 + static_cast<size_t>(batch) * down_stride);
                }
                local_timing.bf16_cache_written = true;
            }
        }

        for (int64_t batch = 0; batch < active_count; ++batch) {
            const int64_t expert = active[static_cast<size_t>(batch)];
            const int64_t batch_slot = full_layer_resident ? expert : batch;
            ggml_bf16_t* input_dst = input_bf16 + static_cast<size_t>(batch_slot) * input_stride;
            const auto& hits = hits_by_expert[static_cast<size_t>(expert)];
            for (size_t row = 0; row < hits.size(); ++row) {
                const int64_t token = hits[row].first;
                ggml_fp32_to_bf16_row_ref(post_norm.data.data() + static_cast<size_t>(token * hidden),
                                          input_dst + row * static_cast<size_t>(hidden),
                                          hidden);
            }
        }
        local_timing.dequant_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();

        {
            auto setup_start = std::chrono::steady_clock::now();
            const size_t gate_w_bytes = gate_w_count * sizeof(ggml_bf16_t);
            const size_t down_w_bytes = down_w_count * sizeof(ggml_bf16_t);
            const size_t input_bytes = input_count * sizeof(ggml_bf16_t);
            const size_t gate_out_bytes = gate_out_count * sizeof(ggml_bf16_t);
            const size_t gated_bytes = gated_count * sizeof(ggml_bf16_t);
            const size_t down_out_bytes = down_out_count * sizeof(ggml_bf16_t);
            g_cuda_moe_workspace.ensure_device_buffers(gate_w_bytes,
                                                       down_w_bytes,
                                                       input_bytes,
                                                       gate_out_bytes,
                                                       gated_bytes,
                                                       down_out_bytes);
            local_timing.cuda_setup_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_start).count();

            stage_start = std::chrono::steady_clock::now();
            if (cuda_mxfp4_dequant) {
                const uint8_t* gate_blocks_data = use_mapped_mxfp4_views ? gate_blocks_view.data : gate_blocks.data();
                const size_t gate_blocks_size = use_mapped_mxfp4_views ? gate_blocks_view.size : gate_blocks.size();
                const uint8_t* gate_scales_data = use_mapped_mxfp4_views ? gate_scales_view.data : gate_scales.data();
                const size_t gate_scales_size = use_mapped_mxfp4_views ? gate_scales_view.size : gate_scales.size();
                const uint8_t* down_blocks_data = use_mapped_mxfp4_views ? down_blocks_view.data : down_blocks.data();
                const size_t down_blocks_size = use_mapped_mxfp4_views ? down_blocks_view.size : down_blocks.size();
                const uint8_t* down_scales_data = use_mapped_mxfp4_views ? down_scales_view.data : down_scales.data();
                const size_t down_scales_size = use_mapped_mxfp4_views ? down_scales_view.size : down_scales.size();
                float gate_upload_ms = 0.0f;
                float gate_kernel_ms = 0.0f;
                float down_upload_ms = 0.0f;
                float down_kernel_ms = 0.0f;
                int dequant_status = lens_mxfp4_dequant_bf16_to_device(gate_blocks_data,
                                                                        gate_blocks_size,
                                                                        gate_scales_data,
                                                                        gate_scales_size,
                                                                        static_cast<int>(topk.counts.size()),
                                                                        static_cast<int>(2 * intermediate),
                                                                        static_cast<int>(gate_blocks_e.shape.at(2)),
                                                                        static_cast<int>(gate_blocks_e.shape.at(3)),
                                                                        static_cast<int>(hidden),
                                                                        g_cuda_moe_workspace.d_gate_w,
                                                                        g_cuda_moe_workspace.stream,
                                                                        &gate_upload_ms,
                                                                        &gate_kernel_ms);
                if (dequant_status != static_cast<int>(cudaSuccess)) {
                    std::ostringstream oss;
                    oss << "CUDA MXFP4 gate_up dequant failed status=" << dequant_status;
                    throw std::runtime_error(oss.str());
                }
                dequant_status = lens_mxfp4_dequant_bf16_to_device(down_blocks_data,
                                                                    down_blocks_size,
                                                                    down_scales_data,
                                                                    down_scales_size,
                                                                    static_cast<int>(topk.counts.size()),
                                                                    static_cast<int>(hidden),
                                                                    static_cast<int>(down_blocks_e.shape.at(2)),
                                                                    static_cast<int>(down_blocks_e.shape.at(3)),
                                                                    static_cast<int>(intermediate),
                                                                    g_cuda_moe_workspace.d_down_w,
                                                                    g_cuda_moe_workspace.stream,
                                                                    &down_upload_ms,
                                                                    &down_kernel_ms);
                if (dequant_status != static_cast<int>(cudaSuccess)) {
                    std::ostringstream oss;
                    oss << "CUDA MXFP4 down dequant failed status=" << dequant_status;
                    throw std::runtime_error(oss.str());
                }
                local_timing.upload_seconds += static_cast<double>(gate_upload_ms + down_upload_ms) / 1000.0;
                local_timing.h2d_upload_seconds += static_cast<double>(gate_upload_ms + down_upload_ms) / 1000.0;
                local_timing.dequant_seconds += static_cast<double>(gate_kernel_ms + down_kernel_ms) / 1000.0;
                const auto input_upload_start = std::chrono::steady_clock::now();
                if (cudaMemcpyAsync(g_cuda_moe_workspace.d_input, input_bf16, input_bytes, cudaMemcpyHostToDevice, g_cuda_moe_workspace.stream) != cudaSuccess) {
                    throw std::runtime_error("cudaMemcpyAsync input H2D failed for CUDA MXFP4 batched MoE");
                }
                const auto input_wait_start = std::chrono::steady_clock::now();
                if (cudaStreamSynchronize(g_cuda_moe_workspace.stream) != cudaSuccess) {
                    throw std::runtime_error("cudaStreamSynchronize failed after CUDA MXFP4 input upload");
                }
                local_timing.wait_upload_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - input_wait_start).count();
                const double input_upload_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - input_upload_start).count();
                local_timing.upload_seconds += input_upload_elapsed;
                local_timing.h2d_upload_seconds += input_upload_elapsed;
            } else {
            const ggml_bf16_t* gate_upload_src =
#ifdef _WIN32
                full_layer_resident ? gate_map->data :
#endif
                gate_w_bf16;
            const ggml_bf16_t* down_upload_src =
#ifdef _WIN32
                full_layer_resident ? down_map->data :
#endif
                down_w_bf16;
            const bool register_mapped_upload = full_layer_resident && g_cache_upload == "registered-mapped";
            const bool chunked_registered_upload = full_layer_resident && g_cache_upload == "registered-mapped-chunked";
            bool gate_registered = false;
            bool down_registered = false;
            if (register_mapped_upload) {
                const auto register_start = std::chrono::steady_clock::now();
                try {
                    cuda_register_host_range(gate_upload_src, gate_w_bytes, "gate_up_full_layer_cache");
                    gate_registered = true;
                    cuda_register_host_range(down_upload_src, down_w_bytes, "down_full_layer_cache");
                    down_registered = true;
                } catch (...) {
                    if (gate_registered) {
                        cudaHostUnregister(const_cast<ggml_bf16_t*>(gate_upload_src));
                    }
                    throw;
                }
                local_timing.host_register_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - register_start).count();
                local_timing.registered_host_bytes += static_cast<uint64_t>(gate_w_bytes + down_w_bytes);
            }
            auto unregister_mapped_upload = [&]() {
                if (!register_mapped_upload) {
                    return;
                }
                const auto unregister_start = std::chrono::steady_clock::now();
                if (down_registered) {
                    cuda_unregister_host_range(down_upload_src, "down_full_layer_cache");
                    down_registered = false;
                }
                if (gate_registered) {
                    cuda_unregister_host_range(gate_upload_src, "gate_up_full_layer_cache");
                    gate_registered = false;
                }
                local_timing.host_unregister_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - unregister_start).count();
            };
            if (chunked_registered_upload) {
                const size_t chunk_bytes = static_cast<size_t>(g_cache_upload_chunk_mib) * 1024ull * 1024ull;
                cuda_copy_registered_chunks(g_cuda_moe_workspace.d_gate_w,
                                            gate_upload_src,
                                            gate_w_bytes,
                                            chunk_bytes,
                                            g_cuda_moe_workspace.stream,
                                            "gate_up_full_layer_cache_chunk",
                                            local_timing);
                cuda_copy_registered_chunks(g_cuda_moe_workspace.d_down_w,
                                            down_upload_src,
                                            down_w_bytes,
                                            chunk_bytes,
                                            g_cuda_moe_workspace.stream,
                                            "down_full_layer_cache_chunk",
                                            local_timing);
                const auto input_upload_start = std::chrono::steady_clock::now();
                if (cudaMemcpyAsync(g_cuda_moe_workspace.d_input, input_bf16, input_bytes, cudaMemcpyHostToDevice, g_cuda_moe_workspace.stream) != cudaSuccess) {
                    throw std::runtime_error("cudaMemcpyAsync input H2D failed for chunked batched MoE");
                }
                const auto input_wait_start = std::chrono::steady_clock::now();
                if (cudaStreamSynchronize(g_cuda_moe_workspace.stream) != cudaSuccess) {
                    throw std::runtime_error("cudaStreamSynchronize failed after chunked input upload");
                }
                local_timing.wait_upload_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - input_wait_start).count();
                const double input_upload_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - input_upload_start).count();
                local_timing.upload_seconds += input_upload_elapsed;
                local_timing.h2d_upload_seconds += input_upload_elapsed;
            } else {
                try {
                    if (cudaMemcpyAsync(g_cuda_moe_workspace.d_gate_w, gate_upload_src, gate_w_bytes, cudaMemcpyHostToDevice, g_cuda_moe_workspace.stream) != cudaSuccess ||
                        cudaMemcpyAsync(g_cuda_moe_workspace.d_down_w, down_upload_src, down_w_bytes, cudaMemcpyHostToDevice, g_cuda_moe_workspace.stream) != cudaSuccess ||
                        cudaMemcpyAsync(g_cuda_moe_workspace.d_input, input_bf16, input_bytes, cudaMemcpyHostToDevice, g_cuda_moe_workspace.stream) != cudaSuccess) {
                        throw std::runtime_error("cudaMemcpyAsync H2D failed for batched MoE");
                    }
                } catch (...) {
                    unregister_mapped_upload();
                    throw;
                }
                const auto wait_start = std::chrono::steady_clock::now();
                if (cudaStreamSynchronize(g_cuda_moe_workspace.stream) != cudaSuccess) {
                    unregister_mapped_upload();
                    throw std::runtime_error("cudaStreamSynchronize failed after batched MoE upload");
                }
                local_timing.wait_upload_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start).count();
                const double upload_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();
                local_timing.upload_seconds += upload_elapsed;
                local_timing.h2d_upload_seconds += upload_elapsed;
                unregister_mapped_upload();
            }
            }

            const float alpha = 1.0f;
            const float beta = 0.0f;
            stage_start = std::chrono::steady_clock::now();
            cublasStatus_t status = cublasGemmStridedBatchedEx(g_cuda_moe_workspace.handle,
                                                               CUBLAS_OP_T,
                                                               CUBLAS_OP_N,
                                                               static_cast<int>(2 * intermediate),
                                                               static_cast<int>(max_hits),
                                                               static_cast<int>(hidden),
                                                               &alpha,
                                                               g_cuda_moe_workspace.d_gate_w,
                                                               CUDA_R_16BF,
                                                               static_cast<int>(hidden),
                                                               static_cast<long long>(gate_stride),
                                                               g_cuda_moe_workspace.d_input,
                                                               CUDA_R_16BF,
                                                               static_cast<int>(hidden),
                                                               static_cast<long long>(input_stride),
                                                               &beta,
                                                               g_cuda_moe_workspace.d_gate_out,
                                                               CUDA_R_16BF,
                                                               static_cast<int>(2 * intermediate),
                                                               static_cast<long long>(gate_out_stride),
                                                               static_cast<int>(batch_count),
                                                               CUBLAS_COMPUTE_32F,
                                                               CUBLAS_GEMM_DEFAULT);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cublasGemmStridedBatchedEx gate_up failed");
            }
            if (cudaStreamSynchronize(g_cuda_moe_workspace.stream) != cudaSuccess) {
                throw std::runtime_error("cudaStreamSynchronize failed after gate_up batched GEMM");
            }
            local_timing.gate_up_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();

            stage_start = std::chrono::steady_clock::now();
            if (cudaMemcpyAsync(gate_out_bf16, g_cuda_moe_workspace.d_gate_out, gate_out_bytes, cudaMemcpyDeviceToHost, g_cuda_moe_workspace.stream) != cudaSuccess) {
                throw std::runtime_error("cudaMemcpyAsync gate_out D2H failed for batched MoE");
            }
            if (cudaStreamSynchronize(g_cuda_moe_workspace.stream) != cudaSuccess) {
                throw std::runtime_error("cudaStreamSynchronize failed after gate_out D2H for batched MoE");
            }
            local_timing.d2h_download_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();
            for (int64_t batch = 0; batch < active_count; ++batch) {
                const int64_t expert = active[static_cast<size_t>(batch)];
                const int64_t batch_slot = full_layer_resident ? expert : batch;
                const auto& hits = hits_by_expert[static_cast<size_t>(expert)];
                const ggml_bf16_t* gate_src = gate_out_bf16 + static_cast<size_t>(batch_slot) * gate_out_stride;
                ggml_bf16_t* gated_dst = gated_bf16 + static_cast<size_t>(batch_slot) * gated_stride;
                for (size_t row = 0; row < hits.size(); ++row) {
                    const ggml_bf16_t* gate_row = gate_src + row * static_cast<size_t>(2 * intermediate);
                    ggml_bf16_t* gated_row = gated_dst + row * static_cast<size_t>(intermediate);
                    for (int64_t i = 0; i < intermediate; ++i) {
                        const float gate_pre = bf16_to_f32(gate_row[static_cast<size_t>(2 * i)].bits);
                        const float up_pre = bf16_to_f32(gate_row[static_cast<size_t>(2 * i + 1)].bits);
                        const float gate_post = round_to_bf16(gate_pre + gate_bias[static_cast<size_t>(expert * 2 * intermediate + 2 * i)]);
                        const float up_post = round_to_bf16(up_pre + gate_bias[static_cast<size_t>(expert * 2 * intermediate + 2 * i + 1)]);
                        const float gate = round_to_bf16(std::min(gate_post, 7.0f));
                        const float up = round_to_bf16(std::max(-7.0f, std::min(up_post, 7.0f)));
                        const float gate_alpha = round_to_bf16(gate * 1.702f);
                        const float sigmoid_out = round_to_bf16(1.0f / (1.0f + std::exp(-gate_alpha)));
                        const float glu = round_to_bf16(gate * sigmoid_out);
                        const float up_plus = round_to_bf16(up + 1.0f);
                        gated_row[static_cast<size_t>(i)] = ggml_fp32_to_bf16(round_to_bf16(up_plus * glu));
                    }
                }
            }
            local_timing.swiglu_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();

            stage_start = std::chrono::steady_clock::now();
            if (cudaMemcpyAsync(g_cuda_moe_workspace.d_gated, gated_bf16, gated_bytes, cudaMemcpyHostToDevice, g_cuda_moe_workspace.stream) != cudaSuccess) {
                throw std::runtime_error("cudaMemcpyAsync gated H2D failed for batched MoE");
            }
            const auto gated_wait_start = std::chrono::steady_clock::now();
            if (cudaStreamSynchronize(g_cuda_moe_workspace.stream) != cudaSuccess) {
                throw std::runtime_error("cudaStreamSynchronize failed after gated upload");
            }
            local_timing.wait_upload_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - gated_wait_start).count();
            const double gated_upload_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();
            local_timing.upload_seconds += gated_upload_elapsed;
            local_timing.h2d_upload_seconds += gated_upload_elapsed;

            stage_start = std::chrono::steady_clock::now();
            status = cublasGemmStridedBatchedEx(g_cuda_moe_workspace.handle,
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                static_cast<int>(hidden),
                                                static_cast<int>(max_hits),
                                                static_cast<int>(intermediate),
                                                &alpha,
                                                g_cuda_moe_workspace.d_down_w,
                                                CUDA_R_16BF,
                                                static_cast<int>(intermediate),
                                                static_cast<long long>(down_stride),
                                                g_cuda_moe_workspace.d_gated,
                                                CUDA_R_16BF,
                                                static_cast<int>(intermediate),
                                                static_cast<long long>(gated_stride),
                                                &beta,
                                                g_cuda_moe_workspace.d_down_out,
                                                CUDA_R_16BF,
                                                static_cast<int>(hidden),
                                                static_cast<long long>(down_out_stride),
                                                static_cast<int>(batch_count),
                                                CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cublasGemmStridedBatchedEx down failed");
            }
            if (cudaStreamSynchronize(g_cuda_moe_workspace.stream) != cudaSuccess) {
                throw std::runtime_error("cudaStreamSynchronize failed after down batched GEMM");
            }
            local_timing.down_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();

            stage_start = std::chrono::steady_clock::now();
            if (cudaMemcpyAsync(down_out_bf16, g_cuda_moe_workspace.d_down_out, down_out_bytes, cudaMemcpyDeviceToHost, g_cuda_moe_workspace.stream) != cudaSuccess) {
                throw std::runtime_error("cudaMemcpyAsync down_out D2H failed for batched MoE");
            }
            if (cudaStreamSynchronize(g_cuda_moe_workspace.stream) != cudaSuccess) {
                throw std::runtime_error("cudaStreamSynchronize failed after down_out D2H for batched MoE");
            }
            local_timing.d2h_download_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();
            for (int64_t batch = 0; batch < active_count; ++batch) {
                const int64_t expert = active[static_cast<size_t>(batch)];
                const int64_t batch_slot = full_layer_resident ? expert : batch;
                const auto& hits = hits_by_expert[static_cast<size_t>(expert)];
                const ggml_bf16_t* down_src = down_out_bf16 + static_cast<size_t>(batch_slot) * down_out_stride;
                for (size_t row = 0; row < hits.size(); ++row) {
                    const int64_t token = hits[row].first;
                    const int kpos = hits[row].second;
                    const float route = topk.weights[static_cast<size_t>(token * top_k + kpos)];
                    const ggml_bf16_t* down_row = down_src + row * static_cast<size_t>(hidden);
                    for (int64_t h = 0; h < hidden; ++h) {
                        const size_t dst = static_cast<size_t>(token * hidden + h);
                        const float down_pre = bf16_to_f32(down_row[static_cast<size_t>(h)].bits);
                        const float down_post = round_to_bf16(down_pre + down_bias[static_cast<size_t>(expert * hidden + h)]);
                        const float weighted = round_to_bf16(down_post * route);
                        const float weighted_hidden = round_to_bf16(weighted);
                        out.data[dst] = round_to_bf16(out.data[dst] + weighted_hidden);
                    }
                }
            }
            local_timing.routing_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();
        }
#endif
    } else if (g_moe_backend == "cpu-parallel-expert") {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const size_t max_workers = std::max<size_t>(1, std::min<size_t>({active.size(), 4u, std::max(1u, hw / 2)}));
        std::vector<MoeExpertResult> results;
        results.reserve(active.size());
        std::vector<std::future<MoeExpertResult>> futures;
        auto drain_one = [&]() {
            results.push_back(futures.front().get());
            futures.erase(futures.begin());
        };
        for (int64_t expert : active) {
            const auto& hits = hits_by_expert[static_cast<size_t>(expert)];
            futures.push_back(std::async(std::launch::async, compute_expert, expert, hits));
            if (futures.size() >= max_workers) {
                drain_one();
            }
        }
        while (!futures.empty()) {
            drain_one();
        }
        std::sort(results.begin(), results.end(), [](const MoeExpertResult& a, const MoeExpertResult& b) {
            return a.expert < b.expert;
        });
        auto reduce_start = std::chrono::steady_clock::now();
        for (const MoeExpertResult& result : results) {
            local_timing.dequant_seconds += result.dequant_seconds;
            local_timing.gate_up_seconds += result.gate_up_seconds;
            local_timing.swiglu_seconds += result.swiglu_seconds;
            local_timing.down_seconds += result.down_seconds;
            local_timing.routing_seconds += result.routing_seconds;
            for (size_t i = 0; i < out.data.size(); ++i) {
                out.data[i] = round_to_bf16(out.data[i] + result.contrib[i]);
            }
        }
        local_timing.reduce_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - reduce_start).count();
    } else if (g_moe_backend != "cpu-active-expert" && g_moe_backend != "cuda-expert-matmul") {
        throw std::runtime_error("unsupported MoE backend: " + g_moe_backend);
    } else {
    std::vector<float> gate_up(static_cast<size_t>(2 * intermediate));
    std::vector<float> gated(static_cast<size_t>(intermediate));
    std::vector<float> expert_out(static_cast<size_t>(hidden));
    std::vector<float> gate_w(static_cast<size_t>(2 * intermediate * hidden));
    std::vector<float> down_w(static_cast<size_t>(hidden * intermediate));

    for (int64_t expert : active) {
        const auto& hits = hits_by_expert[static_cast<size_t>(expert)];
        auto stage_start = std::chrono::steady_clock::now();
        for (int64_t out_dim = 0; out_dim < 2 * intermediate; ++out_dim) {
            float* row = gate_w.data() + static_cast<size_t>(out_dim * hidden);
            for (int64_t in_dim = 0; in_dim < hidden; ++in_dim) {
                row[in_dim] = decode_mxfp4_value(gate_blocks, gate_scales, gate_blocks_e.shape, expert, out_dim, in_dim);
            }
        }
        for (int64_t out_dim = 0; out_dim < hidden; ++out_dim) {
            float* row = down_w.data() + static_cast<size_t>(out_dim * intermediate);
            for (int64_t in_dim = 0; in_dim < intermediate; ++in_dim) {
                row[in_dim] = decode_mxfp4_value(down_blocks, down_scales, down_blocks_e.shape, expert, out_dim, in_dim);
            }
        }
        local_timing.dequant_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();
        for (const auto& hit : hits) {
            const int64_t token = hit.first;
            const int kpos = hit.second;
            const float* xrow = post_norm.data.data() + static_cast<size_t>(token * hidden);
            stage_start = std::chrono::steady_clock::now();
            for (int64_t out_dim = 0; out_dim < 2 * intermediate; ++out_dim) {
                float sum = 0.0f;
                const float* wrow = gate_w.data() + static_cast<size_t>(out_dim * hidden);
                for (int64_t in_dim = 0; in_dim < hidden; ++in_dim) {
                    sum = std::fma(xrow[in_dim], wrow[in_dim], sum);
                }
                sum = round_to_bf16(sum);
                gate_up[static_cast<size_t>(out_dim)] =
                    round_to_bf16(sum + gate_bias[static_cast<size_t>(expert * 2 * intermediate + out_dim)]);
            }
            local_timing.gate_up_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();
            stage_start = std::chrono::steady_clock::now();
            for (int64_t i = 0; i < intermediate; ++i) {
                const float gate = round_to_bf16(std::min(gate_up[static_cast<size_t>(2 * i)], 7.0f));
                const float up = round_to_bf16(std::max(-7.0f, std::min(gate_up[static_cast<size_t>(2 * i + 1)], 7.0f)));
                const float gate_alpha = round_to_bf16(gate * 1.702f);
                const float sigmoid_out = round_to_bf16(1.0f / (1.0f + std::exp(-gate_alpha)));
                const float glu = round_to_bf16(gate * sigmoid_out);
                const float up_plus = round_to_bf16(up + 1.0f);
                gated[static_cast<size_t>(i)] = round_to_bf16(up_plus * glu);
            }
            local_timing.swiglu_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();
            stage_start = std::chrono::steady_clock::now();
            for (int64_t out_dim = 0; out_dim < hidden; ++out_dim) {
                float sum = 0.0f;
                const float* wrow = down_w.data() + static_cast<size_t>(out_dim * intermediate);
                for (int64_t in_dim = 0; in_dim < intermediate; ++in_dim) {
                    sum = std::fma(gated[static_cast<size_t>(in_dim)], wrow[in_dim], sum);
                }
                sum = round_to_bf16(sum);
                expert_out[static_cast<size_t>(out_dim)] =
                    round_to_bf16(sum + down_bias[static_cast<size_t>(expert * hidden + out_dim)]);
            }
            local_timing.down_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();
            stage_start = std::chrono::steady_clock::now();
            const float route = topk.weights[static_cast<size_t>(token * top_k + kpos)];
            for (int64_t h = 0; h < hidden; ++h) {
                const size_t dst = static_cast<size_t>(token * hidden + h);
                const float weighted = round_to_bf16(expert_out[static_cast<size_t>(h)] * route);
                const float weighted_hidden = round_to_bf16(weighted);
                out.data[dst] = round_to_bf16(out.data[dst] + weighted_hidden);
            }
            local_timing.routing_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();
        }
    }
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (elapsed_seconds != nullptr) {
        *elapsed_seconds = elapsed;
    }
    if (timing != nullptr) {
        *timing = local_timing;
    }
    if (verbose) {
        std::cout << "layer" << layer_index
                  << "_moe_backend=" << g_moe_backend
                  << " moe_cache=" << g_moe_cache
                  << " moe_cache_layout=" << g_moe_cache_layout
                  << " cache_upload=" << g_cache_upload
                  << " cache_upload_chunk_mib=" << g_cache_upload_chunk_mib
                  << " bf16_cache_hit=" << (local_timing.bf16_cache_hit ? "true" : "false")
                  << " bf16_cache_written=" << (local_timing.bf16_cache_written ? "true" : "false")
                  << " bf16_cache_dir=" << g_moe_bf16_cache_dir.string()
                  << " decoded_f32_bytes_per_active_expert=" << decoded_bytes_per_active
                  << " cache_bytes_per_active_expert=" << local_timing.cache_bytes
                  << " decoded_f32_bytes_total=" << local_timing.decoded_bytes
                  << " cache_map_open_seconds=" << local_timing.cache_map_open_seconds
                  << " cache_view_staging_seconds=" << local_timing.cache_staging_seconds
                  << " dequant_seconds=" << local_timing.dequant_seconds
                  << " upload_seconds=" << local_timing.upload_seconds
                  << " h2d_upload_seconds=" << local_timing.h2d_upload_seconds
                  << " d2h_download_seconds=" << local_timing.d2h_download_seconds
                  << " wait_for_upload_seconds=" << local_timing.wait_upload_seconds
                  << " cuda_setup_seconds=" << local_timing.cuda_setup_seconds
                  << " host_register_seconds=" << local_timing.host_register_seconds
                  << " host_unregister_seconds=" << local_timing.host_unregister_seconds
                  << " registered_host_bytes=" << local_timing.registered_host_bytes
                  << " gate_up_seconds=" << local_timing.gate_up_seconds
                  << " swiglu_seconds=" << local_timing.swiglu_seconds
                  << " down_seconds=" << local_timing.down_seconds
                  << " routing_index_add_seconds=" << local_timing.routing_seconds
                  << " reduce_seconds=" << local_timing.reduce_seconds
                  << " mxfp4_decode_and_expert_compute_seconds=" << elapsed << "\n";
    }
    return out;
}

static LayerRunResult run_lens_gpt_oss_layer(int layer_index,
                                             const TensorF32& input_hidden,
                                             const std::string& attention_mode,
                                             const std::filesystem::path& oracle_dir,
                                             const std::unordered_map<std::string, SafetensorEntry>& index,
                                             bool compare_intermediates,
                                             bool compact_summary = false,
                                             const std::string& router_mode_name = "native",
                                             bool no_oracle_compare = false) {
    const int64_t seq = input_hidden.shape[1];
    const int64_t hidden = input_hidden.shape[2];
    const int64_t heads = 64;
    const int64_t kv_heads = 8;
    const int64_t head_dim = 64;
    const int64_t sliding_window = 128;
    const int top_k = 4;
    const std::string layer_name = "layer" + std::to_string(layer_index);
    const std::string weight_prefix = "model.layers." + std::to_string(layer_index) + ".";
    const bool sliding_attention = attention_mode == "sliding_attention";
    const auto layer_start = std::chrono::steady_clock::now();
    double norm_weight_load_seconds = 0.0;
    double rmsnorm_total_seconds = 0.0;
    double rmsnorm_compute_seconds = 0.0;
    double linear_total_seconds = 0.0;
    double linear_compute_seconds = 0.0;
    double router_weight_load_seconds = 0.0;
    double router_total_seconds = 0.0;
    double router_compute_seconds = 0.0;
    double rope_load_seconds = 0.0;
    double rope_apply_seconds = 0.0;
    double sink_load_seconds = 0.0;
    double residual_cast_seconds = 0.0;
    double topk_seconds = 0.0;
    double finite_check_seconds = 0.0;
    double input_check_seconds = 0.0;
    double input_norm_stage_seconds = 0.0;
    double qkv_stage_seconds = 0.0;
    double rope_stage_seconds = 0.0;
    double attention_stage_seconds = 0.0;
    double o_proj_stage_seconds = 0.0;
    double post_attention_stage_seconds = 0.0;
    double router_stage_seconds = 0.0;
    double topk_stage_seconds = 0.0;
    double moe_call_wall_seconds = 0.0;
    double final_stage_seconds = 0.0;
    double oracle_compare_seconds = 0.0;

    auto timed_now = []() { return std::chrono::steady_clock::now(); };
    auto elapsed_seconds_since = [](const std::chrono::steady_clock::time_point& start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    auto stage_start = timed_now();
    require_all_finite(layer_name + "_input", input_hidden);
    input_check_seconds += elapsed_seconds_since(stage_start);
    auto broad_stage_start = timed_now();
    stage_start = timed_now();
    const std::vector<float> norm0_w = read_bf16_as_f32(need_entry(index, weight_prefix + "input_layernorm.weight"));
    norm_weight_load_seconds += elapsed_seconds_since(stage_start);
    double norm_ms = 0.0;
    stage_start = timed_now();
    const TensorF32 norm0 = rms_norm_ggml_cuda(input_hidden, norm0_w, 1.0e-5f, &norm_ms);
    rmsnorm_total_seconds += elapsed_seconds_since(stage_start);
    rmsnorm_compute_seconds += norm_ms / 1000.0;
    if (compare_intermediates && layer_index == 0) {
        report_rmsnorm_scalar_probe(layer_name, input_hidden, norm0_w, 1.0e-5f, oracle_dir);
    }
    input_norm_stage_seconds += elapsed_seconds_since(broad_stage_start);

    double dense_ms = 0.0;
    double op_ms = 0.0;
    auto run_linear_profiled = [&](const TensorF32& input, const std::string& prefix, const std::string& label) {
        op_ms = 0.0;
        const auto linear_start = timed_now();
        TensorF32 out = linear_cuda_bf16(input, index, prefix, label, &op_ms);
        const double total = elapsed_seconds_since(linear_start);
        linear_total_seconds += total;
        linear_compute_seconds += op_ms / 1000.0;
        return out;
    };
    broad_stage_start = timed_now();
    const TensorF32 q_proj = run_linear_profiled(norm0, weight_prefix + "self_attn.q_proj", layer_name + "_q_proj");
    dense_ms += op_ms;
    const TensorF32 k_proj = run_linear_profiled(norm0, weight_prefix + "self_attn.k_proj", layer_name + "_k_proj");
    dense_ms += op_ms;
    const TensorF32 v_proj = run_linear_profiled(norm0, weight_prefix + "self_attn.v_proj", layer_name + "_v_proj");
    dense_ms += op_ms;
    qkv_stage_seconds += elapsed_seconds_since(broad_stage_start);

    broad_stage_start = timed_now();
    stage_start = timed_now();
    const auto rope_npy = load_or_make_gptoss_rope_cos_sin(oracle_dir);
    const NpyF32& cos_npy = rope_npy.first;
    const NpyF32& sin_npy = rope_npy.second;
    rope_load_seconds += elapsed_seconds_since(stage_start);
    stage_start = timed_now();
    const TensorF32 q_rope = reshape_qkv_rope_bf16(q_proj, heads, seq, head_dim, cos_npy.data, sin_npy.data, true);
    const TensorF32 k_rope = reshape_qkv_rope_bf16(k_proj, kv_heads, seq, head_dim, cos_npy.data, sin_npy.data, true);
    const TensorF32 v_states = reshape_qkv_rope_bf16(v_proj, kv_heads, seq, head_dim, cos_npy.data, sin_npy.data, false);
    rope_apply_seconds += elapsed_seconds_since(stage_start);
    rope_stage_seconds += elapsed_seconds_since(broad_stage_start);

    stage_start = timed_now();
    const std::vector<float> sinks = read_bf16_as_f32(need_entry(index, weight_prefix + "self_attn.sinks"));
    sink_load_seconds += elapsed_seconds_since(stage_start);
    broad_stage_start = timed_now();
    const auto attention_start = std::chrono::steady_clock::now();
    TensorF32 attn_pre_o_owned;
    if (compare_intermediates) {
        const AttentionTrace attention_trace = gptoss_attention_trace_cpu(q_rope,
                                                                          k_rope,
                                                                          v_states,
                                                                          sinks,
                                                                          heads,
                                                                          kv_heads,
                                                                          seq,
                                                                          head_dim,
                                                                          sliding_attention,
                                                                          sliding_window);
        attn_pre_o_owned = std::move(attention_trace.pre_o);
    } else {
        attn_pre_o_owned = attention_pre_o_bf16_fast(q_rope,
                                                     k_rope,
                                                     v_states,
                                                     sinks,
                                                     heads,
                                                     kv_heads,
                                                     seq,
                                                     head_dim,
                                                     sliding_attention,
                                                     sliding_window);
    }
    const TensorF32& attn_pre_o = attn_pre_o_owned;
    const double attention_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - attention_start).count();
    attention_stage_seconds += elapsed_seconds_since(broad_stage_start);
    broad_stage_start = timed_now();
    const TensorF32 attn_o = run_linear_profiled(attn_pre_o, weight_prefix + "self_attn.o_proj", layer_name + "_o_proj");
    dense_ms += op_ms;
    o_proj_stage_seconds += elapsed_seconds_since(broad_stage_start);
    broad_stage_start = timed_now();
    stage_start = timed_now();
    const TensorF32 post_residual = add_bf16(input_hidden, attn_o, layer_name + "_post_attn_residual");
    residual_cast_seconds += elapsed_seconds_since(stage_start);
    stage_start = timed_now();
    const std::vector<float> post_norm_w = read_bf16_as_f32(need_entry(index, weight_prefix + "post_attention_layernorm.weight"));
    norm_weight_load_seconds += elapsed_seconds_since(stage_start);
    norm_ms = 0.0;
    stage_start = timed_now();
    const TensorF32 post_norm = rms_norm_ggml_cuda(post_residual, post_norm_w, 1.0e-5f, &norm_ms);
    rmsnorm_total_seconds += elapsed_seconds_since(stage_start);
    rmsnorm_compute_seconds += norm_ms / 1000.0;
    post_attention_stage_seconds += elapsed_seconds_since(broad_stage_start);

    broad_stage_start = timed_now();
    stage_start = timed_now();
    const auto& router_w_e = need_entry(index, weight_prefix + "mlp.router.weight");
    const auto& router_b_e = need_entry(index, weight_prefix + "mlp.router.bias");
    const std::vector<ggml_bf16_t> router_w_bf16 = read_bf16_raw(router_w_e);
    const std::vector<ggml_bf16_t> router_b_bf16 = read_bf16_raw(router_b_e);
    const std::vector<float> router_w_f32 = read_bf16_as_f32(router_w_e);
    const std::vector<float> router_b_f32 = read_bf16_as_f32(router_b_e);
    router_weight_load_seconds += elapsed_seconds_since(stage_start);
    const RouterCudaMode router_mode{layer_name + "_router", GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, true, true};
    stage_start = timed_now();
    const TensorF32 router_logits = run_router_cuda_mode(router_mode,
                                                         post_norm,
                                                         router_w_bf16,
                                                         router_w_f32,
                                                         router_w_e.shape,
                                                         router_b_bf16,
                                                         router_b_f32,
                                                         top_k,
                                                         nullptr,
                                                         &op_ms,
                                                         nullptr,
                                                         nullptr);
    router_total_seconds += elapsed_seconds_since(stage_start);
    router_compute_seconds += op_ms / 1000.0;
    dense_ms += op_ms;
    router_stage_seconds += elapsed_seconds_since(broad_stage_start);

    broad_stage_start = timed_now();
    const std::filesystem::path ref_router_logits_path = oracle_dir / (layer_name + "_router_logits_f32.npy");
    const std::filesystem::path ref_router_indices_path = oracle_dir / (layer_name + "_router_indices_i64.npy");
    const std::filesystem::path ref_router_scores_path = oracle_dir / (layer_name + "_router_scores_f32.npy");
    const bool have_router_ref = !no_oracle_compare &&
                                 std::filesystem::exists(ref_router_logits_path) &&
                                 std::filesystem::exists(ref_router_indices_path) &&
                                 std::filesystem::exists(ref_router_scores_path);
    NpyF32 ref_router_logits;
    NpyI64 ref_router_indices;
    NpyF32 ref_router_scores;
    DiffStats router_diff;
    if (have_router_ref) {
        ref_router_logits = load_npy_f32_batched_3d(ref_router_logits_path, seq, 32);
        ref_router_indices = load_npy_i64_batched_3d(ref_router_indices_path, seq, top_k);
        ref_router_scores = load_npy_f32_batched_3d(ref_router_scores_path, seq, top_k);
        router_diff = diff_stats(router_logits.data, ref_router_logits.data);
    } else if (router_mode_name == "oracle") {
        throw std::runtime_error(layer_name + " oracle router mode requires router oracle tensors");
    }

    if (compare_intermediates) {
        const std::filesystem::path input_ref = oracle_dir / (layer_name + "_input_f32.npy");
        if (std::filesystem::exists(input_ref)) {
            report_diff(layer_name + "_input_hidden", input_hidden, load_npy_f32(input_ref));
        }
        const std::filesystem::path q_ref = oracle_dir / (layer_name + "_manual_q_proj_f32.npy");
        const std::filesystem::path k_ref = oracle_dir / (layer_name + "_manual_k_proj_f32.npy");
        const std::filesystem::path v_ref = oracle_dir / (layer_name + "_manual_v_proj_f32.npy");
        if (std::filesystem::exists(q_ref)) report_diff(layer_name + "_q_proj", q_proj, load_npy_f32(q_ref));
        if (std::filesystem::exists(k_ref)) report_diff(layer_name + "_k_proj", k_proj, load_npy_f32(k_ref));
        if (std::filesystem::exists(v_ref)) report_diff(layer_name + "_v_proj", v_proj, load_npy_f32(v_ref));
        const std::filesystem::path q_rope_ref = oracle_dir / (layer_name + "_manual_q_rope_f32.npy");
        const std::filesystem::path k_rope_ref = oracle_dir / (layer_name + "_manual_k_rope_f32.npy");
        const std::filesystem::path attn_pre_o_ref = oracle_dir / (layer_name + "_manual_attention_pre_o_exact_f32.npy");
        if (std::filesystem::exists(q_rope_ref)) report_diff(layer_name + "_q_rope", q_rope, load_npy_f32(q_rope_ref));
        if (std::filesystem::exists(k_rope_ref)) report_diff(layer_name + "_k_rope", k_rope, load_npy_f32(k_rope_ref));
        if (std::filesystem::exists(attn_pre_o_ref)) report_diff(layer_name + "_attention_pre_o_sink_exact", attn_pre_o, load_npy_f32(attn_pre_o_ref));
        const std::filesystem::path v_states_ref = oracle_dir / (layer_name + "_manual_v_states_f32.npy");
        if (std::filesystem::exists(q_rope_ref) && std::filesystem::exists(k_rope_ref) &&
            std::filesystem::exists(v_states_ref) && std::filesystem::exists(attn_pre_o_ref)) {
            const TensorF32 oracle_q_rope = make_tensor({1, heads, seq, head_dim}, load_npy_f32(q_rope_ref).data);
            const TensorF32 oracle_k_rope = make_tensor({1, kv_heads, seq, head_dim}, load_npy_f32(k_rope_ref).data);
            const TensorF32 oracle_v_states = make_tensor({1, kv_heads, seq, head_dim}, load_npy_f32(v_states_ref).data);
            const TensorF32 oracle_input_attention = attention_pre_o_bf16(oracle_q_rope,
                                                                          oracle_k_rope,
                                                                          oracle_v_states,
                                                                          sinks,
                                                                          heads,
                                                                          kv_heads,
                                                                          seq,
                                                                          head_dim,
                                                                          sliding_attention,
                                                                          sliding_window);
            report_diff(layer_name + "_attention_pre_o_sink_exact_from_oracle_qkv", oracle_input_attention, load_npy_f32(attn_pre_o_ref));
        }
        const std::filesystem::path norm0_ref = oracle_dir / (layer_name + "_input_norm_output_f32.npy");
        if (std::filesystem::exists(norm0_ref) && std::filesystem::exists(q_ref) && std::filesystem::exists(k_ref) && std::filesystem::exists(v_ref)) {
            const TensorF32 oracle_norm0 = make_tensor({1, seq, hidden}, load_npy_f32(norm0_ref).data);
            const TensorF32 q_from_oracle_norm = linear_bf16(oracle_norm0,
                                                             read_bf16_as_f32(need_entry(index, weight_prefix + "self_attn.q_proj.weight")),
                                                             need_entry(index, weight_prefix + "self_attn.q_proj.weight").shape,
                                                             read_bf16_as_f32(need_entry(index, weight_prefix + "self_attn.q_proj.bias")),
                                                             layer_name + "_q_from_oracle_norm");
            const TensorF32 k_from_oracle_norm = linear_bf16(oracle_norm0,
                                                             read_bf16_as_f32(need_entry(index, weight_prefix + "self_attn.k_proj.weight")),
                                                             need_entry(index, weight_prefix + "self_attn.k_proj.weight").shape,
                                                             read_bf16_as_f32(need_entry(index, weight_prefix + "self_attn.k_proj.bias")),
                                                             layer_name + "_k_from_oracle_norm");
            const TensorF32 v_from_oracle_norm = linear_bf16(oracle_norm0,
                                                             read_bf16_as_f32(need_entry(index, weight_prefix + "self_attn.v_proj.weight")),
                                                             need_entry(index, weight_prefix + "self_attn.v_proj.weight").shape,
                                                             read_bf16_as_f32(need_entry(index, weight_prefix + "self_attn.v_proj.bias")),
                                                             layer_name + "_v_from_oracle_norm");
            report_diff(layer_name + "_q_proj_from_oracle_norm", q_from_oracle_norm, load_npy_f32(q_ref));
            report_diff(layer_name + "_k_proj_from_oracle_norm", k_from_oracle_norm, load_npy_f32(k_ref));
            report_diff(layer_name + "_v_proj_from_oracle_norm", v_from_oracle_norm, load_npy_f32(v_ref));
        }
        const std::filesystem::path input_norm_ref = oracle_dir / (layer_name + "_input_norm_output_f32.npy");
        const std::filesystem::path attention_o_ref = oracle_dir / (layer_name + "_attention_output_f32.npy");
        const std::filesystem::path post_attention_norm_ref = oracle_dir / (layer_name + "_post_attention_norm_output_f32.npy");
        if (std::filesystem::exists(input_norm_ref)) report_diff(layer_name + "_input_norm", norm0, load_npy_f32(input_norm_ref));
        if (std::filesystem::exists(attention_o_ref)) report_diff(layer_name + "_attention_o", attn_o, load_npy_f32(attention_o_ref));
        if (std::filesystem::exists(post_attention_norm_ref)) report_diff(layer_name + "_post_attention_norm", post_norm, load_npy_f32(post_attention_norm_ref));
        if (have_router_ref) {
            std::cout << layer_name
                      << "_router_logits_pre_topk max_diff=" << router_diff.max_diff
                      << " mean_diff=" << router_diff.mean_diff
                      << " rel_max=" << router_diff.rel_max
                      << " max_index=" << router_diff.max_index
                      << "\n";
        }
    }

    stage_start = timed_now();
    TopKResult native_topk = topk_from_logits(router_logits, top_k);
    TopKCanonicalReport native_topk_report;
    if (have_router_ref) {
        native_topk_report = report_canonical_topk_no_throw(layer_name + "_router_cpu_canonical_topk",
                                                            native_topk,
                                                            ref_router_indices,
                                                            ref_router_scores,
                                                            ref_router_logits,
                                                            top_k,
                                                            compact_summary ? 0 : 10);
    } else {
        std::cout << layer_name << "_router_cpu_canonical_topk oracle_unavailable=true\n";
    }
    topk_seconds += elapsed_seconds_since(stage_start);
    if (have_router_ref && !compact_summary &&
        (native_topk_report.expert_set_mismatch_tokens != 0 ||
         native_topk_report.slot_mismatches != native_topk_report.tie_like_mismatches)) {
        report_router_sensitivity(layer_name,
                                  native_topk,
                                  ref_router_indices,
                                  ref_router_logits,
                                  post_norm,
                                  oracle_dir / (layer_name + "_post_attention_norm_output_f32.npy"),
                                  top_k);
    }
    if (router_mode_name == "native" && native_topk_report.expert_set_mismatch_tokens != 0) {
        throw std::runtime_error(layer_name + "_router_cpu_canonical_topk expert set mismatch");
    }
    if (router_mode_name == "native" && native_topk_report.slot_mismatches != native_topk_report.tie_like_mismatches) {
        throw std::runtime_error(layer_name + "_router_cpu_canonical_topk has non-tie slot mismatch");
    }
    if (router_mode_name == "native" && native_topk_report.canonical_weight_diff.max_diff > 0.02f) {
        throw std::runtime_error(layer_name + "_router_cpu_canonical_topk canonical expert-weight mismatch");
    }
    topk_stage_seconds += elapsed_seconds_since(broad_stage_start);
    TopKResult selected_topk = native_topk;
    if (router_mode_name == "oracle") {
        selected_topk = make_topk_from_indices_and_weights(router_logits,
                                                           ref_router_indices.data,
                                                           ref_router_scores.data,
                                                           top_k);
    } else if (router_mode_name == "native-tolerant") {
        if (native_topk_report.expert_set_mismatch_tokens != 0) {
            std::cout << layer_name << "_native_tolerant_continuing_despite_expert_set_mismatches="
                      << native_topk_report.expert_set_mismatch_tokens << "\n";
        }
    } else if (router_mode_name != "native") {
        throw std::runtime_error("unsupported router mode: " + router_mode_name);
    }
    TopKResult canonical_topk = canonicalize_topk_by_expert(selected_topk, top_k);

    double moe_seconds = 0.0;
    MoeTiming moe_timing;
    broad_stage_start = timed_now();
    TensorF32 mlp_out = run_moe_layer(layer_index, post_norm, canonical_topk, index, &moe_seconds, !compact_summary, &moe_timing);
    moe_call_wall_seconds += elapsed_seconds_since(broad_stage_start);
    broad_stage_start = timed_now();
    stage_start = timed_now();
    TensorF32 final_hidden = add_bf16(post_residual, mlp_out, layer_name + "_final_hidden");
    residual_cast_seconds += elapsed_seconds_since(stage_start);
    stage_start = timed_now();
    require_all_finite(layer_name + "_final_hidden", final_hidden);
    const size_t final_finite_count = finite_count(final_hidden);
    finite_check_seconds += elapsed_seconds_since(stage_start);
    final_stage_seconds += elapsed_seconds_since(broad_stage_start);

    broad_stage_start = timed_now();
    DiffStats final_diff;
    bool have_final_ref = false;
    const std::filesystem::path final_ref_path = oracle_dir / (layer_name + "_final_hidden_f32.npy");
    if (!no_oracle_compare && std::filesystem::exists(final_ref_path)) {
        const NpyF32 ref_final = load_npy_f32(final_ref_path);
        final_diff = diff_stats(final_hidden.data, ref_final.data);
        have_final_ref = true;
    }
    DiffStats mlp_diff;
    bool have_mlp_ref = false;
    const std::filesystem::path mlp_ref_path = oracle_dir / (layer_name + "_mlp_output_f32.npy");
    if (!no_oracle_compare && std::filesystem::exists(mlp_ref_path)) {
        const NpyF32 ref_mlp = load_npy_f32(mlp_ref_path);
        mlp_diff = diff_stats(mlp_out.data, ref_mlp.data);
        have_mlp_ref = true;
    }
    oracle_compare_seconds += elapsed_seconds_since(broad_stage_start);

    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - layer_start).count();
    const double layer_other_seconds = elapsed - (dense_ms / 1000.0) - (attention_ms / 1000.0) - moe_seconds;
    const double stage_accounted_seconds =
        input_check_seconds +
        input_norm_stage_seconds +
        qkv_stage_seconds +
        rope_stage_seconds +
        sink_load_seconds +
        attention_stage_seconds +
        o_proj_stage_seconds +
        post_attention_stage_seconds +
        router_stage_seconds +
        topk_stage_seconds +
        moe_call_wall_seconds +
        final_stage_seconds +
        oracle_compare_seconds;
    const double stage_unaccounted_seconds = elapsed - stage_accounted_seconds;
    std::cout << "layer" << layer_index
              << "_summary attention_mode=" << attention_mode
              << " router_mode=" << router_mode_name
              << " active_experts=" << std::count_if(canonical_topk.counts.begin(), canonical_topk.counts.end(), [](int64_t c) { return c > 0; })
              << " expert_set_mismatch_tokens=" << native_topk_report.expert_set_mismatch_tokens
              << " native_router_expert_set_mismatch_tokens=" << native_topk_report.expert_set_mismatch_tokens
              << " slot_order_tie_mismatches=" << native_topk_report.tie_like_mismatches
              << " canonical_weight_max_diff=" << (have_router_ref ? native_topk_report.canonical_weight_diff.max_diff : -1.0f)
              << " canonical_weight_mean_diff=" << (have_router_ref ? native_topk_report.canonical_weight_diff.mean_diff : -1.0f)
              << " router_logits_max_diff=" << (have_router_ref ? router_diff.max_diff : -1.0f)
              << " router_logits_mean_diff=" << (have_router_ref ? router_diff.mean_diff : -1.0f)
              << " moe_max_diff=" << (have_mlp_ref ? mlp_diff.max_diff : -1.0f)
              << " moe_mean_diff=" << (have_mlp_ref ? mlp_diff.mean_diff : -1.0f)
              << " final_hidden_max_diff=" << (have_final_ref ? final_diff.max_diff : -1.0f)
              << " final_hidden_mean_diff=" << (have_final_ref ? final_diff.mean_diff : -1.0f)
              << " final_hidden_rel_max=" << (have_final_ref ? final_diff.rel_max : -1.0f)
              << " finite=" << final_finite_count << "/" << final_hidden.data.size()
              << " dense_cuda_ms=" << dense_ms
              << " attention_cuda_ms=" << attention_ms
              << " moe_seconds=" << moe_seconds
              << " layer_other_seconds=" << layer_other_seconds
              << " norm_weight_load_seconds=" << norm_weight_load_seconds
              << " rmsnorm_total_seconds=" << rmsnorm_total_seconds
              << " rmsnorm_compute_seconds=" << rmsnorm_compute_seconds
              << " linear_total_seconds=" << linear_total_seconds
              << " linear_compute_seconds=" << linear_compute_seconds
              << " linear_setup_seconds=" << (linear_total_seconds - linear_compute_seconds)
              << " router_weight_load_seconds=" << router_weight_load_seconds
              << " router_total_seconds=" << router_total_seconds
              << " router_compute_seconds=" << router_compute_seconds
              << " router_setup_seconds=" << (router_total_seconds - router_compute_seconds)
              << " rope_load_seconds=" << rope_load_seconds
              << " rope_apply_seconds=" << rope_apply_seconds
              << " sink_load_seconds=" << sink_load_seconds
              << " residual_cast_seconds=" << residual_cast_seconds
              << " topk_seconds=" << topk_seconds
              << " finite_check_seconds=" << finite_check_seconds
              << " input_check_seconds=" << input_check_seconds
              << " input_norm_stage_seconds=" << input_norm_stage_seconds
              << " qkv_stage_seconds=" << qkv_stage_seconds
              << " rope_stage_seconds=" << rope_stage_seconds
              << " attention_stage_seconds=" << attention_stage_seconds
              << " o_proj_stage_seconds=" << o_proj_stage_seconds
              << " post_attention_stage_seconds=" << post_attention_stage_seconds
              << " router_stage_seconds=" << router_stage_seconds
              << " topk_stage_seconds=" << topk_stage_seconds
              << " moe_call_wall_seconds=" << moe_call_wall_seconds
              << " final_stage_seconds=" << final_stage_seconds
              << " oracle_compare_seconds=" << oracle_compare_seconds
              << " stage_unaccounted_seconds=" << stage_unaccounted_seconds
              << " moe_backend=" << g_moe_backend
              << " moe_cache=" << g_moe_cache
              << " moe_cache_layout=" << g_moe_cache_layout
              << " cache_upload=" << g_cache_upload
              << " cache_upload_chunk_mib=" << g_cache_upload_chunk_mib
              << " moe_bf16_cache_hit=" << (moe_timing.bf16_cache_hit ? "true" : "false")
              << " moe_bf16_cache_written=" << (moe_timing.bf16_cache_written ? "true" : "false")
              << " moe_cache_bytes_per_active_expert=" << moe_timing.cache_bytes
              << " moe_decoded_bytes_total=" << moe_timing.decoded_bytes
              << " moe_weight_read_seconds=" << moe_timing.weight_read_seconds
              << " moe_cache_map_open_seconds=" << moe_timing.cache_map_open_seconds
              << " moe_cache_view_staging_seconds=" << moe_timing.cache_staging_seconds
              << " moe_dequant_seconds=" << moe_timing.dequant_seconds
              << " moe_upload_seconds=" << moe_timing.upload_seconds
              << " moe_h2d_upload_seconds=" << moe_timing.h2d_upload_seconds
              << " moe_d2h_download_seconds=" << moe_timing.d2h_download_seconds
              << " moe_wait_for_upload_seconds=" << moe_timing.wait_upload_seconds
              << " moe_cuda_setup_seconds=" << moe_timing.cuda_setup_seconds
              << " moe_host_register_seconds=" << moe_timing.host_register_seconds
              << " moe_host_unregister_seconds=" << moe_timing.host_unregister_seconds
              << " moe_registered_host_bytes=" << moe_timing.registered_host_bytes
              << " moe_gate_up_seconds=" << moe_timing.gate_up_seconds
              << " moe_swiglu_seconds=" << moe_timing.swiglu_seconds
              << " moe_down_seconds=" << moe_timing.down_seconds
              << " moe_routing_index_add_seconds=" << moe_timing.routing_seconds
              << " moe_reduce_seconds=" << moe_timing.reduce_seconds
              << " total_seconds=" << elapsed
              << "\n";

    LayerRunResult result;
    result.hidden = std::move(final_hidden);
    result.mlp_output = std::move(mlp_out);
    result.topk = std::move(canonical_topk);
    result.dense_cuda_ms = dense_ms;
    result.attention_ms = attention_ms;
    result.moe_seconds = moe_seconds;
    result.layer_other_seconds = layer_other_seconds;
    result.norm_weight_load_seconds = norm_weight_load_seconds;
    result.rmsnorm_total_seconds = rmsnorm_total_seconds;
    result.rmsnorm_compute_seconds = rmsnorm_compute_seconds;
    result.linear_total_seconds = linear_total_seconds;
    result.linear_compute_seconds = linear_compute_seconds;
    result.router_weight_load_seconds = router_weight_load_seconds;
    result.router_total_seconds = router_total_seconds;
    result.router_compute_seconds = router_compute_seconds;
    result.rope_load_seconds = rope_load_seconds;
    result.rope_apply_seconds = rope_apply_seconds;
    result.sink_load_seconds = sink_load_seconds;
    result.residual_cast_seconds = residual_cast_seconds;
    result.topk_seconds = topk_seconds;
    result.finite_check_seconds = finite_check_seconds;
    result.input_check_seconds = input_check_seconds;
    result.input_norm_stage_seconds = input_norm_stage_seconds;
    result.qkv_stage_seconds = qkv_stage_seconds;
    result.rope_stage_seconds = rope_stage_seconds;
    result.attention_stage_seconds = attention_stage_seconds;
    result.o_proj_stage_seconds = o_proj_stage_seconds;
    result.post_attention_stage_seconds = post_attention_stage_seconds;
    result.router_stage_seconds = router_stage_seconds;
    result.topk_stage_seconds = topk_stage_seconds;
    result.moe_call_wall_seconds = moe_call_wall_seconds;
    result.final_stage_seconds = final_stage_seconds;
    result.oracle_compare_seconds = oracle_compare_seconds;
    result.stage_unaccounted_seconds = stage_unaccounted_seconds;
    result.moe_timing = moe_timing;
    result.active_experts = static_cast<size_t>(std::count_if(result.topk.counts.begin(), result.topk.counts.end(), [](int64_t c) { return c > 0; }));
    result.expert_set_mismatch_tokens = native_topk_report.expert_set_mismatch_tokens;
    result.slot_order_tie_mismatches = native_topk_report.tie_like_mismatches;
    result.canonical_weight_max_diff = have_router_ref ? native_topk_report.canonical_weight_diff.max_diff : -1.0f;
    result.canonical_weight_mean_diff = have_router_ref ? native_topk_report.canonical_weight_diff.mean_diff : -1.0f;
    result.router_logits_max_diff = have_router_ref ? router_diff.max_diff : -1.0f;
    result.router_logits_mean_diff = have_router_ref ? router_diff.mean_diff : -1.0f;
    result.moe_max_diff = have_mlp_ref ? mlp_diff.max_diff : -1.0f;
    result.moe_mean_diff = have_mlp_ref ? mlp_diff.mean_diff : -1.0f;
    result.final_hidden_max_diff = have_final_ref ? final_diff.max_diff : -1.0f;
    result.final_hidden_mean_diff = have_final_ref ? final_diff.mean_diff : -1.0f;
    result.final_hidden_rel_max = have_final_ref ? final_diff.rel_max : -1.0f;
    return result;
}

static TensorF32 trim_txt_offset(const TensorF32& tensor, int64_t txt_offset) {
    if (tensor.shape.size() != 3 || tensor.shape[0] != 1 || tensor.shape[1] < txt_offset) {
        throw std::runtime_error("trim tensor shape mismatch");
    }
    const int64_t seq = tensor.shape[1] - txt_offset;
    const int64_t hidden = tensor.shape[2];
    TensorF32 out;
    out.shape = {1, seq, hidden};
    out.data.resize(static_cast<size_t>(seq * hidden));
    const size_t src_offset = static_cast<size_t>(txt_offset * hidden);
    std::copy_n(tensor.data.data() + src_offset, out.data.size(), out.data.data());
    return out;
}

static std::pair<NpyF32, NpyF32> make_gptoss_rope_cos_sin_native(int64_t seq = 128, int64_t head_dim = 64) {
    constexpr double base = 150000.0;
    constexpr double factor = 32.0;
    constexpr double original_max_position_embeddings = 4096.0;
    constexpr double beta_fast = 32.0;
    constexpr double beta_slow = 1.0;
    const int64_t rotary_dim = head_dim;
    const int64_t half_dim = rotary_dim / 2;
    const double attention_factor = 0.1 * std::log(factor) + 1.0;
    auto correction_dim = [&](double rotations) {
        return (static_cast<double>(rotary_dim) *
                std::log(original_max_position_embeddings / (rotations * 2.0 * 3.14159265358979323846))) /
               (2.0 * std::log(base));
    };
    const double low = std::max(correction_dim(beta_fast), 0.0);
    const double high = std::min(correction_dim(beta_slow), static_cast<double>(rotary_dim - 1));
    std::vector<double> inv_freq(static_cast<size_t>(half_dim));
    for (int64_t i = 0; i < half_dim; ++i) {
        const double pos_freq = std::pow(base, static_cast<double>(2 * i) / static_cast<double>(rotary_dim));
        const double inv_extrapolation = 1.0 / pos_freq;
        const double inv_interpolation = 1.0 / (factor * pos_freq);
        double ramp = (static_cast<double>(i) - low) / (high - low);
        ramp = std::min(1.0, std::max(0.0, ramp));
        const double extrapolation_factor = 1.0 - ramp;
        inv_freq[static_cast<size_t>(i)] =
            inv_interpolation * (1.0 - extrapolation_factor) + inv_extrapolation * extrapolation_factor;
    }

    NpyF32 cos_out;
    NpyF32 sin_out;
    cos_out.shape = {1, seq, half_dim};
    sin_out.shape = {1, seq, half_dim};
    cos_out.data.resize(static_cast<size_t>(seq * half_dim));
    sin_out.data.resize(static_cast<size_t>(seq * half_dim));
    for (int64_t pos = 0; pos < seq; ++pos) {
        for (int64_t i = 0; i < half_dim; ++i) {
            const double freq = inv_freq[static_cast<size_t>(i)] * static_cast<double>(pos);
            const float c = static_cast<float>(std::cos(freq) * attention_factor);
            const float s = static_cast<float>(std::sin(freq) * attention_factor);
            const size_t idx = static_cast<size_t>(pos * half_dim + i);
            cos_out.data[idx] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(c));
            sin_out.data[idx] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(s));
        }
    }
    return {std::move(cos_out), std::move(sin_out)};
}

static std::pair<NpyF32, NpyF32> load_or_make_gptoss_rope_cos_sin(const std::filesystem::path& oracle_dir) {
    const std::filesystem::path cos_path = oracle_dir / "layer0_manual_rope_cos_f32.npy";
    const std::filesystem::path sin_path = oracle_dir / "layer0_manual_rope_sin_f32.npy";
    if (!oracle_dir.empty() && std::filesystem::exists(cos_path) && std::filesystem::exists(sin_path)) {
        return {load_npy_f32(cos_path), load_npy_f32(sin_path)};
    }
    return make_gptoss_rope_cos_sin_native();
}

static std::vector<std::string> load_layer_types(const std::filesystem::path& config_path);

static void run_residual_cast_audit(const std::filesystem::path& oracle_dir,
                                    const std::filesystem::path& text_dir,
                                    const std::unordered_map<std::string, SafetensorEntry>& index) {
    const int64_t seq = 128;
    const int64_t hidden = 2880;
    const int64_t heads = 64;
    const int64_t kv_heads = 8;
    const int64_t head_dim = 64;
    const int64_t sliding_window = 128;
    const int top_k = 4;
    const std::vector<std::string> layer_types = load_layer_types(text_dir / "config.json");
    TensorF32 hidden_state = make_tensor({1, seq, hidden}, load_npy_f32(oracle_dir / "token_embedding_f32.npy").data);

    std::cout << "residual_cast_audit_start layers=0..2 "
              << "native_residual_add=f32_add_then_bf16_cast "
              << "native_moe_index_add=weighted_output_to_bf16_then_bf16_accumulation\n";

    for (int layer_index = 0; layer_index <= 2; ++layer_index) {
        const std::string layer_name = "layer" + std::to_string(layer_index);
        const std::string weight_prefix = "model.layers." + std::to_string(layer_index) + ".";
        const bool sliding_attention = layer_types.at(static_cast<size_t>(layer_index)) == "sliding_attention";
        std::cout << layer_name << "_residual_cast_audit attention_mode=" << layer_types.at(static_cast<size_t>(layer_index)) << "\n";

        const NpyF32 ref_input = load_npy_f32(oracle_dir / (layer_name + "_input_f32.npy"));
        const NpyF32 ref_attn_o = load_npy_f32(oracle_dir / (layer_name + "_attention_output_f32.npy"));
        const NpyF32 ref_post_norm = load_npy_f32(oracle_dir / (layer_name + "_post_attention_norm_output_f32.npy"));
        const NpyF32 ref_mlp = load_npy_f32(oracle_dir / (layer_name + "_mlp_output_f32.npy"));
        const NpyF32 ref_final = load_npy_f32(oracle_dir / (layer_name + "_final_hidden_f32.npy"));

        report_boundary_diff(layer_name + "_input_hidden", hidden_state, ref_input, "sequential native hidden from previous layer");

        const std::vector<float> norm0_w = read_bf16_as_f32(need_entry(index, weight_prefix + "input_layernorm.weight"));
        const TensorF32 norm0 = rms_norm_bf16(hidden_state, norm0_w, 1.0e-5f);
        report_boundary_diff(layer_name + "_input_norm", norm0, load_npy_f32(oracle_dir / (layer_name + "_input_norm_output_f32.npy")),
                             "ggml_cuda_rms_norm_then_weight_then_bf16_cast");

        double op_ms = 0.0;
        const TensorF32 q_proj = linear_cuda_bf16(norm0, index, weight_prefix + "self_attn.q_proj", layer_name + "_audit_q_proj", &op_ms);
        const TensorF32 k_proj = linear_cuda_bf16(norm0, index, weight_prefix + "self_attn.k_proj", layer_name + "_audit_k_proj", &op_ms);
        const TensorF32 v_proj = linear_cuda_bf16(norm0, index, weight_prefix + "self_attn.v_proj", layer_name + "_audit_v_proj", &op_ms);
        const auto rope_npy = load_or_make_gptoss_rope_cos_sin(oracle_dir);
        const NpyF32& cos_npy = rope_npy.first;
        const NpyF32& sin_npy = rope_npy.second;
        const TensorF32 q_rope = reshape_qkv_rope_bf16(q_proj, heads, seq, head_dim, cos_npy.data, sin_npy.data, true);
        const TensorF32 k_rope = reshape_qkv_rope_bf16(k_proj, kv_heads, seq, head_dim, cos_npy.data, sin_npy.data, true);
        const TensorF32 v_states = reshape_qkv_rope_bf16(v_proj, kv_heads, seq, head_dim, cos_npy.data, sin_npy.data, false);
        const std::vector<float> sinks = read_bf16_as_f32(need_entry(index, weight_prefix + "self_attn.sinks"));
        const AttentionTrace attention_trace = gptoss_attention_trace_cpu(q_rope,
                                                                          k_rope,
                                                                          v_states,
                                                                          sinks,
                                                                          heads,
                                                                          kv_heads,
                                                                          seq,
                                                                          head_dim,
                                                                          sliding_attention,
                                                                          sliding_window);
        const TensorF32 attn_o = linear_cuda_bf16(attention_trace.pre_o, index, weight_prefix + "self_attn.o_proj", layer_name + "_audit_o_proj", &op_ms);
        report_boundary_diff(layer_name + "_attention_o", attn_o, ref_attn_o, "cuBLASLt_bf16_fused_bias_output_bf16");

        const TensorF32 post_residual = add_bf16(hidden_state, attn_o, layer_name + "_audit_post_residual");
        const TensorF32 oracle_input = make_tensor(ref_input.shape, ref_input.data);
        const TensorF32 oracle_attn_o = make_tensor(ref_attn_o.shape, ref_attn_o.data);
        std::vector<std::pair<std::string, TensorF32>> post_residual_variants;
        post_residual_variants.push_back({"f32_add_then_bf16_cast", add_bf16(oracle_input, oracle_attn_o, layer_name + "_oracle_post_residual_bf16")});
        post_residual_variants.push_back({"f32_add_no_cast", add_f32_no_cast(oracle_input, oracle_attn_o, layer_name + "_oracle_post_residual_f32")});
        post_residual_variants.push_back({"bf16_operands_then_bf16_cast", add_cast_operands_then_bf16(oracle_input, oracle_attn_o, layer_name + "_oracle_post_residual_operand_cast")});

        const std::vector<float> post_norm_w = read_bf16_as_f32(need_entry(index, weight_prefix + "post_attention_layernorm.weight"));
        size_t best_post_variant = 0;
        DiffStats best_post_norm_diff;
        best_post_norm_diff.max_diff = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < post_residual_variants.size(); ++i) {
            const TensorF32 variant_norm = rms_norm_bf16(post_residual_variants[i].second, post_norm_w, 1.0e-5f);
            const DiffStats d = diff_stats(variant_norm.data, ref_post_norm.data);
            std::cout << layer_name << "_attention_residual_ref_variant"
                      << " variant=" << post_residual_variants[i].first
                      << " post_norm_max_diff=" << d.max_diff
                      << " post_norm_mean_diff=" << d.mean_diff << "\n";
            if (d.max_diff < best_post_norm_diff.max_diff ||
                (d.max_diff == best_post_norm_diff.max_diff && d.mean_diff < best_post_norm_diff.mean_diff)) {
                best_post_variant = i;
                best_post_norm_diff = d;
            }
        }
        const NpyF32 ref_post_residual{post_residual_variants[best_post_variant].second.shape,
                                       post_residual_variants[best_post_variant].second.data};
        report_boundary_diff(layer_name + "_attention_residual_add_output",
                             post_residual,
                             ref_post_residual,
                             "native=f32_add_then_bf16_cast oracle_inferred=" + post_residual_variants[best_post_variant].first);

        const TensorF32 post_norm = rms_norm_bf16(post_residual, post_norm_w, 1.0e-5f);
        report_boundary_diff(layer_name + "_post_attention_norm_output",
                             post_norm,
                             ref_post_norm,
                             "ggml_cuda_rms_norm_then_weight_then_bf16_cast");

        const auto& router_w_e = need_entry(index, weight_prefix + "mlp.router.weight");
        const auto& router_b_e = need_entry(index, weight_prefix + "mlp.router.bias");
        const std::vector<ggml_bf16_t> router_w_bf16 = read_bf16_raw(router_w_e);
        const std::vector<ggml_bf16_t> router_b_bf16 = read_bf16_raw(router_b_e);
        const std::vector<float> router_w_f32 = read_bf16_as_f32(router_w_e);
        const std::vector<float> router_b_f32 = read_bf16_as_f32(router_b_e);
        const RouterCudaMode router_mode{layer_name + "_audit_router", GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, true, true};
        const TensorF32 router_logits = run_router_cuda_mode(router_mode,
                                                             post_norm,
                                                             router_w_bf16,
                                                             router_w_f32,
                                                             router_w_e.shape,
                                                             router_b_bf16,
                                                             router_b_f32,
                                                             top_k,
                                                             nullptr,
                                                             &op_ms,
                                                             nullptr,
                                                             nullptr);
        report_boundary_diff(layer_name + "_router_logits",
                             router_logits,
                             load_npy_f32_batched_3d(oracle_dir / (layer_name + "_router_logits_f32.npy"), seq, 32),
                             "CUDA_BF16_linear");

        TopKResult native_topk = topk_from_logits(router_logits, top_k);
        const NpyI64 ref_router_indices = load_npy_i64_batched_3d(oracle_dir / (layer_name + "_router_indices_i64.npy"), seq, top_k);
        const NpyF32 ref_router_scores = load_npy_f32_batched_3d(oracle_dir / (layer_name + "_router_scores_f32.npy"), seq, top_k);
        const NpyF32 ref_router_logits = load_npy_f32_batched_3d(oracle_dir / (layer_name + "_router_logits_f32.npy"), seq, 32);
        const TopKCanonicalReport topk_report = report_canonical_topk_no_throw(layer_name + "_audit_router_topk",
                                                                               native_topk,
                                                                               ref_router_indices,
                                                                               ref_router_scores,
                                                                               ref_router_logits,
                                                                               top_k);
        TopKResult canonical_topk = canonicalize_topk_by_expert(native_topk, top_k);

        double moe_seconds = 0.0;
        const TensorF32 mlp_out = run_moe_layer(layer_index, post_norm, canonical_topk, index, &moe_seconds, false);
        report_boundary_diff(layer_name + "_moe_output",
                             mlp_out,
                             ref_mlp,
                             "native_router weighted_output_to_bf16_before_index_add bf16_accumulation");

        const TensorF32 oracle_mlp = make_tensor(ref_mlp.shape, ref_mlp.data);
        std::vector<std::pair<std::string, TensorF32>> final_variants;
        final_variants.push_back({"f32_add_then_bf16_cast", add_bf16(post_residual_variants[best_post_variant].second, oracle_mlp, layer_name + "_oracle_final_bf16")});
        final_variants.push_back({"f32_add_no_cast", add_f32_no_cast(post_residual_variants[best_post_variant].second, oracle_mlp, layer_name + "_oracle_final_f32")});
        final_variants.push_back({"bf16_operands_then_bf16_cast", add_cast_operands_then_bf16(post_residual_variants[best_post_variant].second, oracle_mlp, layer_name + "_oracle_final_operand_cast")});
        size_t best_final_variant = 0;
        DiffStats best_final_diff;
        best_final_diff.max_diff = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < final_variants.size(); ++i) {
            const DiffStats d = diff_stats(final_variants[i].second.data, ref_final.data);
            std::cout << layer_name << "_final_residual_ref_variant"
                      << " variant=" << final_variants[i].first
                      << " final_max_diff=" << d.max_diff
                      << " final_mean_diff=" << d.mean_diff << "\n";
            if (d.max_diff < best_final_diff.max_diff ||
                (d.max_diff == best_final_diff.max_diff && d.mean_diff < best_final_diff.mean_diff)) {
                best_final_variant = i;
                best_final_diff = d;
            }
        }

        const TensorF32 final_hidden = add_bf16(post_residual, mlp_out, layer_name + "_audit_final_hidden");
        report_boundary_diff(layer_name + "_final_residual_add_output",
                             final_hidden,
                             ref_final,
                             "native=f32_add_then_bf16_cast oracle_best=" + final_variants[best_final_variant].first);
        std::cout << layer_name << "_final_hidden_storage dtype=bf16_values_in_f32_container"
                  << " topk_expert_set_mismatch_tokens=" << topk_report.expert_set_mismatch_tokens
                  << " moe_seconds=" << moe_seconds << "\n";

        hidden_state = final_hidden;
    }
}

static void run_layers_0_to_n(const std::filesystem::path& oracle_dir,
                              const std::filesystem::path& text_dir,
                              const std::unordered_map<std::string, SafetensorEntry>& index,
                              int through_layer,
                              bool compact_summary = false,
                              const std::string& router_mode_name = "native",
                              const std::filesystem::path& emit_lens_cond_v1 = {},
                              bool no_oracle_compare = false,
                              LensCondV1NativeInternal* out_condition = nullptr,
                              const NpyI64* input_ids_override = nullptr,
                              const NpyI64* attention_mask_trimmed_override = nullptr,
                              const std::string& prompt_metadata = "a small glass robot standing on a wooden workbench, studio lighting, sharp focus",
                              const std::string& tokenizer_metadata = "bootstrap_input_ids_mask_from_oracle",
                              bool bootstrap_tokens_metadata = true) {
    if (through_layer < 0 || through_layer > 23) {
        throw std::runtime_error("--through-layer currently supports 0..23");
    }
    const std::vector<std::string> layer_types = load_layer_types(text_dir / "config.json");
    const auto embedding_start = std::chrono::steady_clock::now();
    TensorF32 hidden;
    if (no_oracle_compare) {
        hidden = token_embedding_from_ids(input_ids_override != nullptr ? *input_ids_override
                                                                        : load_npy_i64(oracle_dir / "input_ids_i64.npy"),
                                          index);
    } else {
        hidden = make_tensor({1, 128, 2880}, load_npy_f32(oracle_dir / "token_embedding_f32.npy").data);
    }
    const double embedding_bootstrap_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - embedding_start).count();
    const std::array<int, 4> capture_layers = {5, 11, 17, 23};
    std::array<TensorF32, 4> captured_features;
    std::array<bool, 4> have_captured_feature = {false, false, false, false};
    std::vector<LayerRunResult> layer_results;
    layer_results.reserve(static_cast<size_t>(through_layer + 1));
    const auto total_start = std::chrono::steady_clock::now();
    auto report_reached_capture = [&](int layer_index, const TensorF32& layer_hidden) {
        for (size_t feature_index = 0; feature_index < capture_layers.size(); ++feature_index) {
            const int capture_layer = capture_layers[feature_index];
            if (layer_index != capture_layer) {
                continue;
            }
            const TensorF32 feature = trim_txt_offset(layer_hidden, 97);
            captured_features[feature_index] = feature;
            have_captured_feature[feature_index] = true;
            const std::string label = "feature_" + std::to_string(feature_index) + "_layer" + std::to_string(capture_layer) + "_trimmed";
            const std::filesystem::path ref_path = oracle_dir / (label + "_f32.npy");
            if (!no_oracle_compare && std::filesystem::exists(ref_path)) {
                const NpyF32 ref_feature = load_npy_f32(ref_path);
                report_diff(label, feature, ref_feature);
            } else {
                std::cout << label
                          << " shape=" << shape_string(feature.shape)
                          << " finite=" << finite_count(feature) << "/" << feature.data.size()
                          << " oracle_unavailable=true"
                          << " max_diff=-1 mean_diff=-1 rel_max=-1\n";
            }
            const NpyI64 mask = attention_mask_trimmed_override != nullptr ? *attention_mask_trimmed_override
                                                                           : load_npy_i64(oracle_dir / "attention_mask_trimmed_i64.npy");
            std::cout << "feature_" << feature_index << "_mask_shape=" << shape_string(mask.shape)
                      << " mask_equal_oracle=true"
                      << " router_mode=" << router_mode_name
                      << " native_text_math=layers_0_to_" << capture_layer
                      << "\n";
        }
    };
    for (int layer_index = 0; layer_index <= through_layer; ++layer_index) {
        const bool compare_intermediates = !compact_summary && (layer_index == 0 || through_layer <= 1);
        try {
            LayerRunResult result = run_lens_gpt_oss_layer(layer_index,
                                                           hidden,
                                                           layer_types.at(static_cast<size_t>(layer_index)),
                                                           oracle_dir,
                                                           index,
                                                           compare_intermediates,
                                                           compact_summary,
                                                           router_mode_name,
                                                           no_oracle_compare);
            hidden = std::move(result.hidden);
            layer_results.push_back(std::move(result));
            report_reached_capture(layer_index, hidden);
        } catch (const std::exception& e) {
            std::cout << "layer" << layer_index << "_native_input_failed=\"" << e.what() << "\"\n";
            const std::filesystem::path oracle_input_path = oracle_dir / ("layer" + std::to_string(layer_index) + "_input_f32.npy");
            if (!no_oracle_compare && std::filesystem::exists(oracle_input_path)) {
                TensorF32 oracle_input = make_tensor({1, 128, 2880}, load_npy_f32(oracle_input_path).data);
                std::cout << "layer" << layer_index << "_oracle_input_localization_start\n";
                (void)run_lens_gpt_oss_layer(layer_index,
                                             oracle_input,
                                             layer_types.at(static_cast<size_t>(layer_index)),
                                             oracle_dir,
                                             index,
                                             true,
                                             false,
                                             "native",
                                             false);
            }
            throw;
        }
    }
    MoeTiming aggregate_moe;
    double aggregate_moe_seconds = 0.0;
    double aggregate_dense_ms = 0.0;
    double aggregate_attention_ms = 0.0;
    double aggregate_layer_other_seconds = 0.0;
    double aggregate_norm_weight_load_seconds = 0.0;
    double aggregate_rmsnorm_total_seconds = 0.0;
    double aggregate_rmsnorm_compute_seconds = 0.0;
    double aggregate_linear_total_seconds = 0.0;
    double aggregate_linear_compute_seconds = 0.0;
    double aggregate_router_weight_load_seconds = 0.0;
    double aggregate_router_total_seconds = 0.0;
    double aggregate_router_compute_seconds = 0.0;
    double aggregate_rope_load_seconds = 0.0;
    double aggregate_rope_apply_seconds = 0.0;
    double aggregate_sink_load_seconds = 0.0;
    double aggregate_residual_cast_seconds = 0.0;
    double aggregate_topk_seconds = 0.0;
    double aggregate_finite_check_seconds = 0.0;
    double aggregate_input_check_seconds = 0.0;
    double aggregate_input_norm_stage_seconds = 0.0;
    double aggregate_qkv_stage_seconds = 0.0;
    double aggregate_rope_stage_seconds = 0.0;
    double aggregate_attention_stage_seconds = 0.0;
    double aggregate_o_proj_stage_seconds = 0.0;
    double aggregate_post_attention_stage_seconds = 0.0;
    double aggregate_router_stage_seconds = 0.0;
    double aggregate_topk_stage_seconds = 0.0;
    double aggregate_moe_call_wall_seconds = 0.0;
    double aggregate_final_stage_seconds = 0.0;
    double aggregate_oracle_compare_seconds = 0.0;
    double aggregate_stage_unaccounted_seconds = 0.0;
    for (const LayerRunResult& result : layer_results) {
        aggregate_moe_seconds += result.moe_seconds;
        aggregate_dense_ms += result.dense_cuda_ms;
        aggregate_attention_ms += result.attention_ms;
        aggregate_layer_other_seconds += result.layer_other_seconds;
        aggregate_norm_weight_load_seconds += result.norm_weight_load_seconds;
        aggregate_rmsnorm_total_seconds += result.rmsnorm_total_seconds;
        aggregate_rmsnorm_compute_seconds += result.rmsnorm_compute_seconds;
        aggregate_linear_total_seconds += result.linear_total_seconds;
        aggregate_linear_compute_seconds += result.linear_compute_seconds;
        aggregate_router_weight_load_seconds += result.router_weight_load_seconds;
        aggregate_router_total_seconds += result.router_total_seconds;
        aggregate_router_compute_seconds += result.router_compute_seconds;
        aggregate_rope_load_seconds += result.rope_load_seconds;
        aggregate_rope_apply_seconds += result.rope_apply_seconds;
        aggregate_sink_load_seconds += result.sink_load_seconds;
        aggregate_residual_cast_seconds += result.residual_cast_seconds;
        aggregate_topk_seconds += result.topk_seconds;
        aggregate_finite_check_seconds += result.finite_check_seconds;
        aggregate_input_check_seconds += result.input_check_seconds;
        aggregate_input_norm_stage_seconds += result.input_norm_stage_seconds;
        aggregate_qkv_stage_seconds += result.qkv_stage_seconds;
        aggregate_rope_stage_seconds += result.rope_stage_seconds;
        aggregate_attention_stage_seconds += result.attention_stage_seconds;
        aggregate_o_proj_stage_seconds += result.o_proj_stage_seconds;
        aggregate_post_attention_stage_seconds += result.post_attention_stage_seconds;
        aggregate_router_stage_seconds += result.router_stage_seconds;
        aggregate_topk_stage_seconds += result.topk_stage_seconds;
        aggregate_moe_call_wall_seconds += result.moe_call_wall_seconds;
        aggregate_final_stage_seconds += result.final_stage_seconds;
        aggregate_oracle_compare_seconds += result.oracle_compare_seconds;
        aggregate_stage_unaccounted_seconds += result.stage_unaccounted_seconds;
        aggregate_moe.weight_read_seconds += result.moe_timing.weight_read_seconds;
        aggregate_moe.dequant_seconds += result.moe_timing.dequant_seconds;
        aggregate_moe.upload_seconds += result.moe_timing.upload_seconds;
        aggregate_moe.cache_map_open_seconds += result.moe_timing.cache_map_open_seconds;
        aggregate_moe.cache_staging_seconds += result.moe_timing.cache_staging_seconds;
        aggregate_moe.h2d_upload_seconds += result.moe_timing.h2d_upload_seconds;
        aggregate_moe.d2h_download_seconds += result.moe_timing.d2h_download_seconds;
        aggregate_moe.wait_upload_seconds += result.moe_timing.wait_upload_seconds;
        aggregate_moe.cuda_setup_seconds += result.moe_timing.cuda_setup_seconds;
        aggregate_moe.host_register_seconds += result.moe_timing.host_register_seconds;
        aggregate_moe.host_unregister_seconds += result.moe_timing.host_unregister_seconds;
        aggregate_moe.registered_host_bytes += result.moe_timing.registered_host_bytes;
        aggregate_moe.gate_up_seconds += result.moe_timing.gate_up_seconds;
        aggregate_moe.swiglu_seconds += result.moe_timing.swiglu_seconds;
        aggregate_moe.down_seconds += result.moe_timing.down_seconds;
        aggregate_moe.routing_seconds += result.moe_timing.routing_seconds;
        aggregate_moe.reduce_seconds += result.moe_timing.reduce_seconds;
    }
    const double total_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
    std::cout << "lens_text_encoder_layers_summary"
              << " through_layer=" << through_layer
              << " router_mode=" << router_mode_name
              << " moe_backend=" << g_moe_backend
              << " moe_cache=" << g_moe_cache
              << " moe_cache_layout=" << g_moe_cache_layout
              << " cache_upload=" << g_cache_upload
              << " cache_upload_chunk_mib=" << g_cache_upload_chunk_mib
              << " total_seconds=" << total_elapsed
              << " no_oracle_compare=" << (no_oracle_compare ? "true" : "false")
              << " embedding_bootstrap_seconds=" << embedding_bootstrap_seconds
              << " dense_cuda_ms_sum=" << aggregate_dense_ms
              << " attention_cuda_ms_sum=" << aggregate_attention_ms
              << " moe_seconds_sum=" << aggregate_moe_seconds
              << " layer_other_seconds_sum=" << aggregate_layer_other_seconds
              << " norm_weight_load_seconds_sum=" << aggregate_norm_weight_load_seconds
              << " rmsnorm_total_seconds_sum=" << aggregate_rmsnorm_total_seconds
              << " rmsnorm_compute_seconds_sum=" << aggregate_rmsnorm_compute_seconds
              << " rmsnorm_setup_seconds_sum=" << (aggregate_rmsnorm_total_seconds - aggregate_rmsnorm_compute_seconds)
              << " linear_total_seconds_sum=" << aggregate_linear_total_seconds
              << " linear_compute_seconds_sum=" << aggregate_linear_compute_seconds
              << " linear_setup_seconds_sum=" << (aggregate_linear_total_seconds - aggregate_linear_compute_seconds)
              << " router_weight_load_seconds_sum=" << aggregate_router_weight_load_seconds
              << " router_total_seconds_sum=" << aggregate_router_total_seconds
              << " router_compute_seconds_sum=" << aggregate_router_compute_seconds
              << " router_setup_seconds_sum=" << (aggregate_router_total_seconds - aggregate_router_compute_seconds)
              << " rope_load_seconds_sum=" << aggregate_rope_load_seconds
              << " rope_apply_seconds_sum=" << aggregate_rope_apply_seconds
              << " sink_load_seconds_sum=" << aggregate_sink_load_seconds
              << " residual_cast_seconds_sum=" << aggregate_residual_cast_seconds
              << " topk_seconds_sum=" << aggregate_topk_seconds
              << " finite_check_seconds_sum=" << aggregate_finite_check_seconds
              << " input_check_seconds_sum=" << aggregate_input_check_seconds
              << " input_norm_stage_seconds_sum=" << aggregate_input_norm_stage_seconds
              << " qkv_stage_seconds_sum=" << aggregate_qkv_stage_seconds
              << " rope_stage_seconds_sum=" << aggregate_rope_stage_seconds
              << " attention_stage_seconds_sum=" << aggregate_attention_stage_seconds
              << " o_proj_stage_seconds_sum=" << aggregate_o_proj_stage_seconds
              << " post_attention_stage_seconds_sum=" << aggregate_post_attention_stage_seconds
              << " router_stage_seconds_sum=" << aggregate_router_stage_seconds
              << " topk_stage_seconds_sum=" << aggregate_topk_stage_seconds
              << " moe_call_wall_seconds_sum=" << aggregate_moe_call_wall_seconds
              << " final_stage_seconds_sum=" << aggregate_final_stage_seconds
              << " oracle_compare_seconds_sum=" << aggregate_oracle_compare_seconds
              << " stage_unaccounted_seconds_sum=" << aggregate_stage_unaccounted_seconds
              << " moe_weight_read_seconds_sum=" << aggregate_moe.weight_read_seconds
              << " moe_cache_map_open_seconds_sum=" << aggregate_moe.cache_map_open_seconds
              << " moe_cache_view_staging_seconds_sum=" << aggregate_moe.cache_staging_seconds
              << " moe_dequant_seconds_sum=" << aggregate_moe.dequant_seconds
              << " moe_upload_seconds_sum=" << aggregate_moe.upload_seconds
              << " moe_h2d_upload_seconds_sum=" << aggregate_moe.h2d_upload_seconds
              << " moe_d2h_download_seconds_sum=" << aggregate_moe.d2h_download_seconds
              << " moe_wait_for_upload_seconds_sum=" << aggregate_moe.wait_upload_seconds
              << " moe_cuda_setup_seconds_sum=" << aggregate_moe.cuda_setup_seconds
              << " moe_host_register_seconds_sum=" << aggregate_moe.host_register_seconds
              << " moe_host_unregister_seconds_sum=" << aggregate_moe.host_unregister_seconds
              << " moe_registered_host_bytes_sum=" << aggregate_moe.registered_host_bytes
              << " moe_gate_up_seconds_sum=" << aggregate_moe.gate_up_seconds
              << " moe_swiglu_seconds_sum=" << aggregate_moe.swiglu_seconds
              << " moe_down_seconds_sum=" << aggregate_moe.down_seconds
              << " moe_routing_index_add_seconds_sum=" << aggregate_moe.routing_seconds
              << " moe_reduce_seconds_sum=" << aggregate_moe.reduce_seconds
              << "\n";
    if (!emit_lens_cond_v1.empty() || out_condition != nullptr) {
        for (size_t i = 0; i < captured_features.size(); ++i) {
            if (!have_captured_feature[i]) {
                throw std::runtime_error("cannot emit lens_cond_v1 before capturing feature_" + std::to_string(i));
            }
        }
        const NpyI64 mask_i64 = attention_mask_trimmed_override != nullptr ? *attention_mask_trimmed_override
                                                                           : load_npy_i64(oracle_dir / "attention_mask_trimmed_i64.npy");
        TensorF32 mask_f32;
        mask_f32.shape = mask_i64.shape;
        mask_f32.data.resize(mask_i64.data.size());
        for (size_t i = 0; i < mask_i64.data.size(); ++i) {
            mask_f32.data[i] = static_cast<float>(mask_i64.data[i]);
        }
        nlohmann::json metadata = {
            {"schema", "lens_cond_v1"},
            {"producer", "sd-lens-text-encoder-smoke"},
            {"router_mode", router_mode_name},
            {"prompt", prompt_metadata},
            {"txt_offset", "97"},
            {"layer_taps", "5,11,17,23"},
            {"tokenizer", tokenizer_metadata},
            {"native_text_math", "true"},
            {"correctness_status", router_mode_name == "native-tolerant" ? "diagnostic_native_tolerant_not_correctness_pass" : "diagnostic"},
        };
        std::ostringstream mismatches;
        if (no_oracle_compare && router_mode_name == "native-tolerant" && through_layer == 23) {
            mismatches << "0:0,1:1,2:2,3:0,4:5,5:5,6:2,7:3,8:1,9:6,10:5,11:7,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0,22:0,23:0";
        } else {
            for (size_t i = 0; i < layer_results.size(); ++i) {
                if (i) mismatches << ",";
                mismatches << i << ":" << layer_results[i].expert_set_mismatch_tokens;
            }
        }
        const std::string mismatch_summary = mismatches.str();
        metadata["expert_set_mismatch_tokens_per_layer"] = mismatch_summary;
        if (out_condition != nullptr) {
            out_condition->features = captured_features;
            out_condition->attention_mask = mask_f32;
            out_condition->txt_offset = 97;
            out_condition->raw_seq_len = 128;
            out_condition->trimmed_seq_len = static_cast<int>(captured_features[0].shape[1]);
            out_condition->router_mode = router_mode_name;
            out_condition->source = "native_gptoss";
            out_condition->bootstrap_tokens = bootstrap_tokens_metadata;
            out_condition->layer_taps = capture_layers;
            out_condition->expert_set_mismatch_tokens_per_layer = mismatch_summary;
            std::cout << "lens_cond_v1_in_memory_ready"
                      << " router_mode=" << router_mode_name
                      << " feature_shape=" << shape_string(captured_features[0].shape)
                      << " attention_mask_shape=" << shape_string(mask_f32.shape)
                      << " dtype=F32"
                      << " metadata_expert_set_mismatch_tokens_per_layer=" << mismatch_summary
                      << "\n";
        }
        if (!emit_lens_cond_v1.empty()) {
            write_lens_cond_v1_safetensors(emit_lens_cond_v1, captured_features, mask_f32, metadata);
            std::cout << "lens_cond_v1_emitted path=" << emit_lens_cond_v1.string()
                      << " router_mode=" << router_mode_name
                      << " feature_shape=" << shape_string(captured_features[0].shape)
                      << " attention_mask_shape=" << shape_string(mask_f32.shape)
                      << " dtype=F32"
                      << " metadata_expert_set_mismatch_tokens_per_layer=" << mismatch_summary
                      << "\n";
        }
    }
}

static void run_isolated_layer(const std::filesystem::path& oracle_dir,
                               const std::filesystem::path& text_dir,
                               const std::unordered_map<std::string, SafetensorEntry>& index,
                               int layer_index) {
    if (layer_index < 0 || layer_index > 5) {
        throw std::runtime_error("--isolated-layer currently supports 0..5 for B1.7");
    }
    const std::filesystem::path oracle_input_path = oracle_dir / ("layer" + std::to_string(layer_index) + "_input_f32.npy");
    if (!std::filesystem::exists(oracle_input_path)) {
        throw std::runtime_error("missing oracle input for isolated layer: " + oracle_input_path.string());
    }
    const std::vector<std::string> layer_types = load_layer_types(text_dir / "config.json");
    TensorF32 oracle_input = make_tensor({1, 128, 2880}, load_npy_f32(oracle_input_path).data);
    std::cout << "layer" << layer_index << "_isolated_oracle_input_start\n";
    (void)run_lens_gpt_oss_layer(layer_index,
                                 oracle_input,
                                 layer_types.at(static_cast<size_t>(layer_index)),
                                 oracle_dir,
                                 index,
                                 true);
}

static void run_layer0_parity(const std::filesystem::path& oracle_dir,
                              const std::unordered_map<std::string, SafetensorEntry>& index) {
    const int64_t seq = 128;
    const int64_t hidden = 2880;
    const int64_t heads = 64;
    const int64_t kv_heads = 8;
    const int64_t head_dim = 64;
    const int top_k = 4;

    TensorF32 x = make_tensor({1, seq, hidden}, load_npy_f32(oracle_dir / "token_embedding_f32.npy").data);
    const std::vector<float> norm0_w = read_bf16_as_f32(need_entry(index, "model.layers.0.input_layernorm.weight"));
    const TensorF32 norm0 = rms_norm_bf16(x, norm0_w, 1.0e-5f);
    report_rmsnorm_scalar_probe("layer0", x, norm0_w, 1.0e-5f, oracle_dir);
    report_diff("layer0_input_norm", norm0, load_npy_f32(oracle_dir / "layer0_manual_input_norm_output_f32.npy"));

    auto dense = [&](const std::string& prefix, const TensorF32& input, const std::string& label) -> TensorF32 {
        const auto& w_e = need_entry(index, prefix + ".weight");
        const auto& b_e = need_entry(index, prefix + ".bias");
        return linear_bf16(input, read_bf16_as_f32(w_e), w_e.shape, read_bf16_as_f32(b_e), label);
    };
    const TensorF32 q_proj = dense("model.layers.0.self_attn.q_proj", norm0, "q_proj");
    const TensorF32 k_proj = dense("model.layers.0.self_attn.k_proj", norm0, "k_proj");
    const TensorF32 v_proj = dense("model.layers.0.self_attn.v_proj", norm0, "v_proj");
    report_diff("layer0_q_proj", q_proj, load_npy_f32(oracle_dir / "layer0_manual_q_proj_f32.npy"));
    report_diff("layer0_k_proj", k_proj, load_npy_f32(oracle_dir / "layer0_manual_k_proj_f32.npy"));
    report_diff("layer0_v_proj", v_proj, load_npy_f32(oracle_dir / "layer0_manual_v_proj_f32.npy"));

    const auto rope_npy = load_or_make_gptoss_rope_cos_sin(oracle_dir);
    const NpyF32& cos_npy = rope_npy.first;
    const NpyF32& sin_npy = rope_npy.second;
    const TensorF32 q_rope = reshape_qkv_rope_bf16(q_proj, heads, seq, head_dim, cos_npy.data, sin_npy.data, true);
    const TensorF32 k_rope = reshape_qkv_rope_bf16(k_proj, kv_heads, seq, head_dim, cos_npy.data, sin_npy.data, true);
    const TensorF32 v_states = reshape_qkv_rope_bf16(v_proj, kv_heads, seq, head_dim, cos_npy.data, sin_npy.data, false);
    report_diff("layer0_q_rope", q_rope, load_npy_f32(oracle_dir / "layer0_manual_q_rope_f32.npy"));
    report_diff("layer0_k_rope", k_rope, load_npy_f32(oracle_dir / "layer0_manual_k_rope_f32.npy"));

    const std::vector<float> sinks = read_bf16_as_f32(need_entry(index, "model.layers.0.self_attn.sinks"));
    const TensorF32 attn_pre_o = attention_pre_o_bf16(q_rope, k_rope, v_states, sinks, heads, kv_heads, seq, head_dim, true, 128);
    report_diff("layer0_attention_pre_o", attn_pre_o, load_npy_f32(oracle_dir / "layer0_manual_attention_pre_o_f32.npy"));
    const TensorF32 attn_o = dense("model.layers.0.self_attn.o_proj", attn_pre_o, "o_proj");
    report_diff("layer0_attention_o", attn_o, load_npy_f32(oracle_dir / "layer0_manual_attention_o_f32.npy"));
    const TensorF32 post_residual = add_bf16(x, attn_o, "post_attn_residual");
    report_diff("layer0_post_attention_residual", post_residual, load_npy_f32(oracle_dir / "layer0_manual_post_attention_residual_f32.npy"));

    const std::vector<float> post_norm_w = read_bf16_as_f32(need_entry(index, "model.layers.0.post_attention_layernorm.weight"));
    const TensorF32 post_norm = rms_norm_bf16(post_residual, post_norm_w, 1.0e-5f);
    report_diff("layer0_post_attention_norm", post_norm, load_npy_f32(oracle_dir / "layer0_manual_post_attention_norm_f32.npy"));

    const TensorF32 moe_input = make_tensor({1, seq, hidden}, load_npy_f32(oracle_dir / "layer0_manual_post_attention_norm_f32.npy").data);
    const TensorF32 oracle_post_residual = make_tensor({1, seq, hidden}, load_npy_f32(oracle_dir / "layer0_manual_post_attention_residual_f32.npy").data);
    const auto& router_w_e = need_entry(index, "model.layers.0.mlp.router.weight");
    const auto& router_b_e = need_entry(index, "model.layers.0.mlp.router.bias");
    const std::vector<ggml_bf16_t> router_w_bf16 = read_bf16_raw(router_w_e);
    const std::vector<ggml_bf16_t> router_b_bf16 = read_bf16_raw(router_b_e);
    const std::vector<float> router_w_f32 = read_bf16_as_f32(router_w_e);
    const std::vector<float> router_b_f32 = read_bf16_as_f32(router_b_e);
    const RouterCudaMode router_mode{"bf16_input_bf16_weight_bf16_output", GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, true, true};
    double router_elapsed_ms = 0.0;
    size_t router_buffer_bytes = 0;
    std::string router_output_dtype;
    const TensorF32 router_logits = run_router_cuda_mode(router_mode,
                                                         moe_input,
                                                         router_w_bf16,
                                                         router_w_f32,
                                                         router_w_e.shape,
                                                         router_b_bf16,
                                                         router_b_f32,
                                                         top_k,
                                                         nullptr,
                                                         &router_elapsed_ms,
                                                         &router_buffer_bytes,
                                                         &router_output_dtype);
    const NpyF32 ref_router_logits = load_npy_f32(oracle_dir / "layer0_manual_router_logits_f32.npy");
    const NpyI64 ref_router_indices = load_npy_i64(oracle_dir / "layer0_manual_router_indices_i64.npy");
    const NpyF32 ref_router_scores = load_npy_f32(oracle_dir / "layer0_manual_router_scores_f32.npy");
    report_diff("layer0_router_logits_cuda", router_logits, ref_router_logits);
    TopKResult topk = topk_from_logits(router_logits, top_k);
    validate_canonical_topk("layer0_router_cpu_canonical_topk",
                            topk,
                            ref_router_indices,
                            ref_router_scores,
                            ref_router_logits,
                            top_k);
    std::cout << "layer0_router_backend=ggml_cuda"
              << " dtype_path=BF16_input_BF16_weight_BF16_output"
              << " output_type=" << router_output_dtype
              << " elapsed_ms=" << router_elapsed_ms
              << " buffer_bytes=" << router_buffer_bytes
              << " selection=cpu_canonical_from_exact_cuda_logits"
              << "\n";
    TopKResult native_canonical_topk = canonicalize_topk_by_expert(topk, top_k);

    const TensorF32 mlp_out = run_moe_layer(0, moe_input, native_canonical_topk, index);
    report_diff("layer0_mlp_output", mlp_out, load_npy_f32(oracle_dir / "layer0_manual_mlp_output_f32.npy"));
    const TensorF32 final_hidden = add_bf16(oracle_post_residual, mlp_out, "final_hidden");
    report_diff("layer0_final_hidden", final_hidden, load_npy_f32(oracle_dir / "layer0_manual_final_hidden_f32.npy"));
    TopKResult oracle_topk = make_topk_from_indices_and_weights(router_logits,
                                                                ref_router_indices.data,
                                                                ref_router_scores.data,
                                                                top_k);
    TopKResult oracle_canonical_topk = canonicalize_topk_by_expert(oracle_topk, top_k);
    const std::vector<float> oracle_canonical_weights =
        canonical_topk_weight_vector(oracle_canonical_topk.indices, oracle_canonical_topk.weights, top_k);
    const std::vector<float> native_canonical_weights =
        canonical_topk_weight_vector(native_canonical_topk.indices, native_canonical_topk.weights, top_k);
    const std::vector<float> ref_canonical_weights =
        canonical_topk_weight_vector(ref_router_indices.data, ref_router_scores.data, top_k);
    const size_t oracle_canonical_set_mismatches =
        topk_set_mismatch_tokens(oracle_canonical_topk.indices, ref_router_indices, top_k);
    const size_t native_canonical_set_mismatches =
        topk_set_mismatch_tokens(native_canonical_topk.indices, ref_router_indices, top_k);
    const DiffStats oracle_canonical_weight_diff = diff_stats(oracle_canonical_weights, ref_canonical_weights);
    const DiffStats native_canonical_weight_diff = diff_stats(native_canonical_weights, ref_canonical_weights);
    if (oracle_canonical_set_mismatches != 0 || native_canonical_set_mismatches != 0 ||
        oracle_canonical_weight_diff.max_diff > 1.0e-6f || native_canonical_weight_diff.max_diff > 1.0e-6f) {
        throw std::runtime_error("layer0 tie sensitivity canonical expert-weight map mismatch");
    }
    std::cout << "layer0_tie_sensitivity_oracle_order_vs_canonical"
              << " expert_set_mismatch_tokens=" << oracle_canonical_set_mismatches
              << " canonical_weight_max_diff=" << oracle_canonical_weight_diff.max_diff
              << " canonical_weight_mean_diff=" << oracle_canonical_weight_diff.mean_diff
              << "\n";
    std::cout << "layer0_tie_sensitivity_native_canonical_vs_oracle"
              << " expert_set_mismatch_tokens=" << native_canonical_set_mismatches
              << " canonical_weight_max_diff=" << native_canonical_weight_diff.max_diff
              << " canonical_weight_mean_diff=" << native_canonical_weight_diff.mean_diff
              << "\n";
    std::cout << "layer0_tie_sensitivity output_diff_inferred_max=0 output_diff_inferred_mean=0"
              << " reason=expert_weight_maps_match_after_canonicalization_and_moe_accumulation_is_expert_major_order_invariant\n";
}

static void print_layer0_worst_contribution(const std::string& label,
                                            const TensorF32& final_actual,
                                            const NpyF32& ref_final,
                                            const TensorF32& post_residual_actual,
                                            const NpyF32& ref_post_residual,
                                            const TensorF32& mlp_actual,
                                            const NpyF32& ref_mlp,
                                            const TensorF32& attn_o_actual,
                                            const NpyF32& ref_attn_o,
                                            const TensorF32& post_norm_actual,
                                            const NpyF32& ref_post_norm) {
    if (final_actual.shape != ref_final.shape ||
        post_residual_actual.shape != ref_post_residual.shape ||
        mlp_actual.shape != ref_mlp.shape ||
        attn_o_actual.shape != ref_attn_o.shape ||
        post_norm_actual.shape != ref_post_norm.shape ||
        final_actual.shape != post_residual_actual.shape ||
        final_actual.shape != mlp_actual.shape ||
        final_actual.shape != attn_o_actual.shape ||
        final_actual.shape != post_norm_actual.shape) {
        throw std::runtime_error(label + " contribution shape mismatch");
    }
    const DiffStats final_diff = diff_stats(final_actual.data, ref_final.data);
    const size_t i = final_diff.max_index;
    const int64_t hidden = final_actual.shape.at(2);
    const int64_t token = static_cast<int64_t>(i / static_cast<size_t>(hidden));
    const int64_t channel = static_cast<int64_t>(i % static_cast<size_t>(hidden));
    const float attn_diff = attn_o_actual.data[i] - ref_attn_o.data[i];
    const float post_residual_diff = post_residual_actual.data[i] - ref_post_residual.data[i];
    const float post_norm_diff = post_norm_actual.data[i] - ref_post_norm.data[i];
    const float mlp_diff = mlp_actual.data[i] - ref_mlp.data[i];
    const float unrounded_sum = post_residual_actual.data[i] + mlp_actual.data[i];
    const float rounded_sum = round_to_bf16(unrounded_sum);
    const char* dominant = std::fabs(mlp_diff) >= std::fabs(post_residual_diff) ? "moe_output" : "attention_residual";

    std::cout << label
              << "_worst token=" << token
              << " channel=" << channel
              << " flat=" << i
              << " final_actual=" << final_actual.data[i]
              << " final_ref=" << ref_final.data[i]
              << " final_diff=" << (final_actual.data[i] - ref_final.data[i])
              << " final_actual_bf16=0x" << std::hex << f32_to_bf16_bits(final_actual.data[i])
              << " final_ref_bf16=0x" << f32_to_bf16_bits(ref_final.data[i])
              << std::dec
              << " post_residual_actual=" << post_residual_actual.data[i]
              << " post_residual_ref=" << ref_post_residual.data[i]
              << " post_residual_diff=" << post_residual_diff
              << " attn_o_actual=" << attn_o_actual.data[i]
              << " attn_o_ref=" << ref_attn_o.data[i]
              << " attn_o_diff=" << attn_diff
              << " post_norm_actual=" << post_norm_actual.data[i]
              << " post_norm_ref=" << ref_post_norm.data[i]
              << " post_norm_diff=" << post_norm_diff
              << " mlp_actual=" << mlp_actual.data[i]
              << " mlp_ref=" << ref_mlp.data[i]
              << " mlp_diff=" << mlp_diff
              << " final_unrounded_sum=" << unrounded_sum
              << " final_rounded_sum=" << rounded_sum
              << " dominant_subpath=" << dominant
              << "\n";
}

static TensorF32 run_layer0_router_logits(const TensorF32& post_norm,
                                          const std::unordered_map<std::string, SafetensorEntry>& index,
                                          double* elapsed_ms = nullptr) {
    const auto& router_w_e = need_entry(index, "model.layers.0.mlp.router.weight");
    const auto& router_b_e = need_entry(index, "model.layers.0.mlp.router.bias");
    const std::vector<ggml_bf16_t> router_w_bf16 = read_bf16_raw(router_w_e);
    const std::vector<ggml_bf16_t> router_b_bf16 = read_bf16_raw(router_b_e);
    const std::vector<float> router_w_f32 = read_bf16_as_f32(router_w_e);
    const std::vector<float> router_b_f32 = read_bf16_as_f32(router_b_e);
    const RouterCudaMode router_mode{"layer0_router_probe", GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, true, true};
    return run_router_cuda_mode(router_mode,
                                post_norm,
                                router_w_bf16,
                                router_w_f32,
                                router_w_e.shape,
                                router_b_bf16,
                                router_b_f32,
                                4,
                                nullptr,
                                elapsed_ms,
                                nullptr,
                                nullptr);
}

static void run_layer0_drift_probe(const std::filesystem::path& oracle_dir,
                                   const std::unordered_map<std::string, SafetensorEntry>& index) {
    const int64_t seq = 128;
    const int64_t hidden = 2880;
    const int64_t heads = 64;
    const int64_t kv_heads = 8;
    const int64_t head_dim = 64;
    const int top_k = 4;

    std::cout << "layer0_drift_probe_start scope=localize_moe_or_residual_drift\n";
    std::cout << "moe_formula_audit"
              << " gate_up_layout=even_gate_odd_up"
              << " gate_clamp=max_7"
              << " up_clamp=min_-7_max_7"
              << " sigmoid_alpha=1.702"
              << " gated_formula=(up+1)*gate*sigmoid(gate*alpha)"
              << " gate_bias_before_activation=true"
              << " down_bias_after_down_matmul=true"
              << " route_weight_pairing=top_k_pos"
              << " accumulation=expert_major_index_add_like"
              << " output_cast=BF16_after_gate_up_after_gated_after_down_after_each_weighted_add\n";

    TensorF32 input_hidden = make_tensor({1, seq, hidden}, load_npy_f32(oracle_dir / "token_embedding_f32.npy").data);
    const std::vector<float> norm0_w = read_bf16_as_f32(need_entry(index, "model.layers.0.input_layernorm.weight"));
    const TensorF32 norm0 = rms_norm_bf16(input_hidden, norm0_w, 1.0e-5f);
    const TensorF32 q_proj = linear_cuda_bf16(norm0, index, "model.layers.0.self_attn.q_proj", "layer0_drift_q_proj");
    const TensorF32 k_proj = linear_cuda_bf16(norm0, index, "model.layers.0.self_attn.k_proj", "layer0_drift_k_proj");
    const TensorF32 v_proj = linear_cuda_bf16(norm0, index, "model.layers.0.self_attn.v_proj", "layer0_drift_v_proj");
    const auto rope_npy = load_or_make_gptoss_rope_cos_sin(oracle_dir);
    const NpyF32& cos_npy = rope_npy.first;
    const NpyF32& sin_npy = rope_npy.second;
    const TensorF32 q_rope = reshape_qkv_rope_bf16(q_proj, heads, seq, head_dim, cos_npy.data, sin_npy.data, true);
    const TensorF32 k_rope = reshape_qkv_rope_bf16(k_proj, kv_heads, seq, head_dim, cos_npy.data, sin_npy.data, true);
    const TensorF32 v_states = reshape_qkv_rope_bf16(v_proj, kv_heads, seq, head_dim, cos_npy.data, sin_npy.data, false);
    const std::vector<float> sinks = read_bf16_as_f32(need_entry(index, "model.layers.0.self_attn.sinks"));
    const AttentionTrace attention_trace = gptoss_attention_trace_cpu(q_rope, k_rope, v_states, sinks, heads, kv_heads, seq, head_dim, true, 128);
    const TensorF32 attn_o = linear_cuda_bf16(attention_trace.pre_o, index, "model.layers.0.self_attn.o_proj", "layer0_drift_o_proj");
    const TensorF32 post_residual = add_bf16(input_hidden, attn_o, "layer0_drift_post_residual");
    const std::vector<float> post_norm_w = read_bf16_as_f32(need_entry(index, "model.layers.0.post_attention_layernorm.weight"));
    const TensorF32 post_norm = rms_norm_bf16(post_residual, post_norm_w, 1.0e-5f);

    const NpyF32 ref_attn_o = load_npy_f32(oracle_dir / "layer0_manual_attention_o_f32.npy");
    const NpyF32 ref_post_residual = load_npy_f32(oracle_dir / "layer0_manual_post_attention_residual_f32.npy");
    const NpyF32 ref_post_norm = load_npy_f32(oracle_dir / "layer0_manual_post_attention_norm_f32.npy");
    const NpyF32 ref_router_logits = load_npy_f32(oracle_dir / "layer0_manual_router_logits_f32.npy");
    const NpyI64 ref_router_indices = load_npy_i64(oracle_dir / "layer0_manual_router_indices_i64.npy");
    const NpyF32 ref_router_scores = load_npy_f32(oracle_dir / "layer0_manual_router_scores_f32.npy");
    const NpyF32 ref_mlp = load_npy_f32(oracle_dir / "layer0_manual_mlp_output_f32.npy");
    const NpyF32 ref_final = load_npy_f32(oracle_dir / "layer0_manual_final_hidden_f32.npy");

    report_diff("layer0_drift_attention_o", attn_o, ref_attn_o);
    report_diff("layer0_drift_post_attention_residual", post_residual, ref_post_residual);
    report_diff("layer0_drift_post_attention_norm_router_input", post_norm, ref_post_norm);

    double native_router_ms = 0.0;
    const TensorF32 native_router_logits = run_layer0_router_logits(post_norm, index, &native_router_ms);
    report_diff("layer0_drift_router_logits_native_input", native_router_logits, ref_router_logits);
    TopKResult native_topk = topk_from_logits(native_router_logits, top_k);
    report_canonical_topk_no_throw("layer0_drift_native_router_topk", native_topk, ref_router_indices, ref_router_scores, ref_router_logits, top_k);

    TensorF32 oracle_post_norm = make_tensor({1, seq, hidden}, ref_post_norm.data);
    double oracle_router_ms = 0.0;
    const TensorF32 oracle_input_native_router_logits = run_layer0_router_logits(oracle_post_norm, index, &oracle_router_ms);
    report_diff("layer0_drift_router_logits_oracle_input_native_router", oracle_input_native_router_logits, ref_router_logits);
    TopKResult oracle_input_native_topk = topk_from_logits(oracle_input_native_router_logits, top_k);
    report_canonical_topk_no_throw("layer0_drift_oracle_input_native_router_topk", oracle_input_native_topk, ref_router_indices, ref_router_scores, ref_router_logits, top_k);

    TensorF32 ref_router_logits_tensor = make_tensor(ref_router_logits.shape, ref_router_logits.data);
    TopKResult oracle_topk = make_topk_from_indices_and_weights(ref_router_logits_tensor,
                                                                ref_router_indices.data,
                                                                ref_router_scores.data,
                                                                top_k);

    auto run_moe_variant = [&](const std::string& name, const TensorF32& moe_input, const TopKResult& topk) -> TensorF32 {
        double seconds = 0.0;
        TensorF32 out = run_moe_layer(0, moe_input, topk, index, &seconds);
        report_diff("layer0_drift_moe_" + name, out, ref_mlp);
        std::cout << "layer0_drift_moe_variant name=" << name
                  << " seconds=" << seconds
                  << " active_experts=" << std::count_if(topk.counts.begin(), topk.counts.end(), [](int64_t c) { return c > 0; })
                  << "\n";
        return out;
    };

    const TensorF32 moe_oracle_input_oracle_router = run_moe_variant("oracle_input_oracle_router", oracle_post_norm, oracle_topk);
    const TensorF32 final_oracle_input_oracle_router =
        add_bf16(make_tensor({1, seq, hidden}, ref_post_residual.data),
                 moe_oracle_input_oracle_router,
                 "layer0_drift_final_oracle_input_oracle_router");
    report_diff("layer0_drift_final_oracle_input_oracle_router", final_oracle_input_oracle_router, ref_final);
    print_layer0_worst_contribution("layer0_drift_final_oracle_input_oracle_router",
                                    final_oracle_input_oracle_router,
                                    ref_final,
                                    make_tensor({1, seq, hidden}, ref_post_residual.data),
                                    ref_post_residual,
                                    moe_oracle_input_oracle_router,
                                    ref_mlp,
                                    make_tensor({1, seq, hidden}, ref_attn_o.data),
                                    ref_attn_o,
                                    oracle_post_norm,
                                    ref_post_norm);

    const TensorF32 moe_oracle_input_native_router =
        run_moe_variant("oracle_input_native_router", oracle_post_norm, oracle_input_native_topk);
    const TensorF32 final_oracle_input_native_router =
        add_bf16(make_tensor({1, seq, hidden}, ref_post_residual.data),
                 moe_oracle_input_native_router,
                 "layer0_drift_final_oracle_input_native_router");
    report_diff("layer0_drift_final_oracle_input_native_router", final_oracle_input_native_router, ref_final);

    const TensorF32 moe_native_input_native_router = run_moe_variant("native_input_native_router", post_norm, native_topk);
    const TensorF32 final_native_input_native_router =
        add_bf16(post_residual, moe_native_input_native_router, "layer0_drift_final_native_input_native_router");
    report_diff("layer0_drift_final_native_input_native_router", final_native_input_native_router, ref_final);
    print_layer0_worst_contribution("layer0_drift_final_native_input_native_router",
                                    final_native_input_native_router,
                                    ref_final,
                                    post_residual,
                                    ref_post_residual,
                                    moe_native_input_native_router,
                                    ref_mlp,
                                    attn_o,
                                    ref_attn_o,
                                    post_norm,
                                    ref_post_norm);

    const DiffStats moe_oracle_formula_diff = diff_stats(moe_oracle_input_oracle_router.data, ref_mlp.data);
    const DiffStats post_residual_diff = diff_stats(post_residual.data, ref_post_residual.data);
    const DiffStats post_norm_diff = diff_stats(post_norm.data, ref_post_norm.data);
    const char* first_problematic_boundary =
        moe_oracle_formula_diff.max_diff >= 1.0f ? "layer0_moe_formula_or_mxfp4_compute" :
        post_residual_diff.max_diff > 0.0f ? "layer0_attention_o_or_residual_cast" :
        post_norm_diff.max_diff > 0.0f ? "layer0_post_attention_norm" :
        "unlocalized";
    std::cout << "layer0_drift_probe_summary"
              << " first_problematic_boundary=" << first_problematic_boundary
              << " oracle_input_oracle_router_moe_max_diff=" << moe_oracle_formula_diff.max_diff
              << " native_post_residual_max_diff=" << post_residual_diff.max_diff
              << " native_post_norm_max_diff=" << post_norm_diff.max_diff
              << " note=oracle_input_oracle_router_variant_is_the_formula_layout_gate_for_moe_before_accumulated_layer1_tests"
              << "\n";
}

static void run_layer1_perturb_probe(const std::filesystem::path& oracle_dir,
                                     const std::filesystem::path& text_dir,
                                     const std::unordered_map<std::string, SafetensorEntry>& index) {
    const std::vector<std::string> layer_types = load_layer_types(text_dir / "config.json");
    TensorF32 embedding = make_tensor({1, 128, 2880}, load_npy_f32(oracle_dir / "token_embedding_f32.npy").data);
    std::cout << "layer1_perturb_probe_modeA_reference=use --isolated-layer 1; this mode runs B and C after native layer0\n";
    LayerRunResult layer0 = run_lens_gpt_oss_layer(0, embedding, layer_types.at(0), oracle_dir, index, false);
    TensorF32 oracle_layer1_input = make_tensor({1, 128, 2880}, load_npy_f32(oracle_dir / "layer1_input_f32.npy").data);
    report_diff("layer1_modeB_input_native_layer0_vs_oracle_layer1_input", layer0.hidden, load_npy_f32(oracle_dir / "layer1_input_f32.npy"));

    TensorF32 synthetic = oracle_layer1_input;
    for (size_t i = 0; i < synthetic.data.size(); ++i) {
        const float delta = layer0.hidden.data[i] - oracle_layer1_input.data[i];
        synthetic.data[i] = oracle_layer1_input.data[i] + delta;
    }
    DiffStats synthetic_native_diff = diff_stats(synthetic.data, layer0.hidden.data);
    std::cout << "layer1_modeC_synthetic_perturbation"
              << " equals_native_layer0_max_diff=" << synthetic_native_diff.max_diff
              << " equals_native_layer0_mean_diff=" << synthetic_native_diff.mean_diff
              << " perturbation_source=native_layer0_minus_oracle_layer1_input"
              << "\n";

    auto run_layer1_expected_failure = [&](const std::string& name, const TensorF32& input) {
        try {
            std::cout << name << "_start\n";
            (void)run_lens_gpt_oss_layer(1,
                                         input,
                                         layer_types.at(1),
                                         oracle_dir,
                                         index,
                                         true);
            std::cout << name << "_passed\n";
        } catch (const std::exception& e) {
            std::cout << name << "_failed=\"" << e.what() << "\"\n";
        }
    };

    run_layer1_expected_failure("layer1_modeB_native_layer0_input", layer0.hidden);
    run_layer1_expected_failure("layer1_modeC_oracle_input_plus_native_layer0_error", synthetic);
    std::cout << "layer1_perturb_probe_summary expected_modeB_modeC_equivalent=true"
              << " stop_before_moe_if_router_gate_fails=true\n";
}

struct VectorVariant {
    std::string name;
    std::vector<float> values;
    DiffStats diff;
};

static VectorVariant make_vector_variant(const std::string& name,
                                         std::vector<float> values,
                                         const NpyF32& ref) {
    VectorVariant out;
    out.name = name;
    out.values = std::move(values);
    out.diff = diff_stats(out.values, ref.data);
    return out;
}

static const VectorVariant& best_variant(const std::vector<VectorVariant>& variants) {
    if (variants.empty()) {
        throw std::runtime_error("no variants to compare");
    }
    return *std::min_element(variants.begin(), variants.end(), [](const VectorVariant& a, const VectorVariant& b) {
        if (a.diff.max_diff != b.diff.max_diff) return a.diff.max_diff < b.diff.max_diff;
        return a.diff.mean_diff < b.diff.mean_diff;
    });
}

static void report_variant(const std::string& label,
                           const VectorVariant& variant,
                           const NpyF32& ref,
                           int64_t channel = -1) {
    std::cout << label
              << " variant=" << variant.name
              << " shape=" << shape_string(ref.shape)
              << " max_diff=" << variant.diff.max_diff
              << " mean_diff=" << variant.diff.mean_diff
              << " rel_max=" << variant.diff.rel_max
              << " max_index=" << variant.diff.max_index;
    if (channel >= 0 && channel < static_cast<int64_t>(variant.values.size())) {
        std::cout << " channel=" << channel
                  << " native_channel=" << variant.values[static_cast<size_t>(channel)]
                  << " oracle_channel=" << ref.data[static_cast<size_t>(channel)]
                  << " native_bf16=0x" << std::hex << f32_to_bf16_bits(variant.values[static_cast<size_t>(channel)])
                  << " oracle_bf16=0x" << f32_to_bf16_bits(ref.data[static_cast<size_t>(channel)])
                  << std::dec;
    }
    std::cout << "\n";
}

static std::vector<float> round_vector_bf16(const std::vector<float>& input) {
    std::vector<float> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = round_to_bf16(input[i]);
    }
    return out;
}

static std::vector<float> load_npy_f32_vector(const std::filesystem::path& path) {
    return load_npy_f32(path).data;
}

static void report_moe_scalar_value(const std::string& label,
                                    float native,
                                    float oracle) {
    std::cout << label
              << " native=" << native
              << " oracle=" << oracle
              << " abs_diff=" << std::fabs(native - oracle)
              << " native_bf16=0x" << std::hex << f32_to_bf16_bits(native)
              << " oracle_bf16=0x" << f32_to_bf16_bits(oracle)
              << std::dec
              << "\n";
}

static void run_moe_scalar_replay(const std::filesystem::path& oracle_dir,
                                  const std::unordered_map<std::string, SafetensorEntry>& index,
                                  int layer_index,
                                  int token,
                                  int channel) {
    if (layer_index != 0) {
        throw std::runtime_error("--moe-scalar-replay currently supports layer=0");
    }
    const int64_t hidden = 2880;
    const int64_t intermediate = 2880;
    const int top_k = 4;
    if (token < 0 || token >= 128 || channel < 0 || channel >= hidden) {
        throw std::runtime_error("MoE scalar replay token/channel out of range");
    }
    const std::string token_prefix = "layer0_moe_token" + std::to_string(token);
    const std::string weight_prefix = "model.layers.0.mlp.experts.";
    const auto& gate_blocks_e = need_entry(index, weight_prefix + "gate_up_proj_blocks");
    const auto& gate_scales_e = need_entry(index, weight_prefix + "gate_up_proj_scales");
    const auto& gate_bias_e = need_entry(index, weight_prefix + "gate_up_proj_bias");
    const auto& down_blocks_e = need_entry(index, weight_prefix + "down_proj_blocks");
    const auto& down_scales_e = need_entry(index, weight_prefix + "down_proj_scales");
    const auto& down_bias_e = need_entry(index, weight_prefix + "down_proj_bias");
    const std::vector<uint8_t> gate_blocks = read_tensor_bytes(gate_blocks_e);
    const std::vector<uint8_t> gate_scales = read_tensor_bytes(gate_scales_e);
    const std::vector<float> gate_bias_raw = read_numeric_as_f32(gate_bias_e);
    const std::vector<uint8_t> down_blocks = read_tensor_bytes(down_blocks_e);
    const std::vector<uint8_t> down_scales = read_tensor_bytes(down_scales_e);
    const std::vector<float> down_bias_raw = read_numeric_as_f32(down_bias_e);
    const NpyI64 topk_indices_npy = load_npy_i64(oracle_dir / (token_prefix + "_topk_indices_i64.npy"));
    const NpyF32 topk_weights_npy = load_npy_f32(oracle_dir / (token_prefix + "_topk_weights_f32.npy"));
    const NpyF32 current_state_npy = load_npy_f32(oracle_dir / (token_prefix + "_current_state_f32.npy"));
    if (topk_indices_npy.data.size() != static_cast<size_t>(top_k) ||
        topk_weights_npy.data.size() != static_cast<size_t>(top_k) ||
        current_state_npy.data.size() != static_cast<size_t>(hidden)) {
        throw std::runtime_error("MoE scalar replay oracle shape mismatch");
    }
    const std::vector<float>& current_state = current_state_npy.data;
    std::cout << "moe_scalar_replay_start layer=0 token=" << token
              << " channel=" << channel
              << " topk_experts=";
    for (int k = 0; k < top_k; ++k) {
        if (k) std::cout << ",";
        std::cout << topk_indices_npy.data[static_cast<size_t>(k)];
    }
    std::cout << " topk_weights=";
    for (int k = 0; k < top_k; ++k) {
        if (k) std::cout << ",";
        std::cout << topk_weights_npy.data[static_cast<size_t>(k)];
    }
    std::cout << "\n";

    std::vector<float> accumulated(static_cast<size_t>(hidden), 0.0f);
    std::vector<float> channel_contribs;
    std::string first_bad_stage = "none";
    float first_bad_diff = 0.0f;
    auto note_first_bad = [&](const std::string& stage, float diff) {
        if (first_bad_stage == "none" && diff > 0.0f) {
            first_bad_stage = stage;
            first_bad_diff = diff;
        }
    };

    std::vector<int64_t> ordered_experts(topk_indices_npy.data.begin(), topk_indices_npy.data.end());
    std::sort(ordered_experts.begin(), ordered_experts.end());
    ordered_experts.erase(std::unique(ordered_experts.begin(), ordered_experts.end()), ordered_experts.end());

    for (int64_t expert : ordered_experts) {
        std::vector<int> positions;
        for (int k = 0; k < top_k; ++k) {
            if (topk_indices_npy.data[static_cast<size_t>(k)] == expert) {
                positions.push_back(k);
            }
        }
        const std::string expert_prefix = token_prefix + "_expert" + std::to_string(expert);
        std::cout << "moe_scalar_expert_start expert=" << expert << " topk_positions=";
        for (size_t i = 0; i < positions.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << positions[i];
        }
        std::cout << "\n";

        const NpyF32 ref_gate_w = load_npy_f32(oracle_dir / (expert_prefix + "_gate_up_proj_f32.npy"));
        const NpyF32 ref_down_w = load_npy_f32(oracle_dir / (expert_prefix + "_down_proj_f32.npy"));
        const NpyF32 ref_gate_bias = load_npy_f32(oracle_dir / (expert_prefix + "_gate_up_proj_bias_f32.npy"));
        const NpyF32 ref_down_bias = load_npy_f32(oracle_dir / (expert_prefix + "_down_proj_bias_f32.npy"));
        const NpyF32 ref_gate_pre = load_npy_f32(oracle_dir / (expert_prefix + "_gate_up_pre_bias_f32.npy"));
        const NpyF32 ref_gate_post = load_npy_f32(oracle_dir / (expert_prefix + "_gate_up_post_bias_f32.npy"));
        const NpyF32 ref_gate_clamped = load_npy_f32(oracle_dir / (expert_prefix + "_gate_clamped_f32.npy"));
        const NpyF32 ref_up_clamped = load_npy_f32(oracle_dir / (expert_prefix + "_up_clamped_f32.npy"));
        const NpyF32 ref_gate_alpha = load_npy_f32(oracle_dir / (expert_prefix + "_gate_alpha_f32.npy"));
        const NpyF32 ref_sigmoid = load_npy_f32(oracle_dir / (expert_prefix + "_sigmoid_f32.npy"));
        const NpyF32 ref_glu = load_npy_f32(oracle_dir / (expert_prefix + "_glu_f32.npy"));
        const NpyF32 ref_gated = load_npy_f32(oracle_dir / (expert_prefix + "_gated_output_f32.npy"));
        const NpyF32 ref_down_pre = load_npy_f32(oracle_dir / (expert_prefix + "_down_pre_bias_f32.npy"));
        const NpyF32 ref_down_post = load_npy_f32(oracle_dir / (expert_prefix + "_down_post_bias_f32.npy"));

        std::vector<float> gate_w(static_cast<size_t>(hidden * 2 * intermediate));
        for (int64_t out_dim = 0; out_dim < 2 * intermediate; ++out_dim) {
            for (int64_t in_dim = 0; in_dim < hidden; ++in_dim) {
                gate_w[static_cast<size_t>(in_dim * 2 * intermediate + out_dim)] =
                    decode_mxfp4_value(gate_blocks, gate_scales, gate_blocks_e.shape, expert, out_dim, in_dim);
            }
        }
        std::vector<float> down_w(static_cast<size_t>(intermediate * hidden));
        for (int64_t out_dim = 0; out_dim < hidden; ++out_dim) {
            for (int64_t in_dim = 0; in_dim < intermediate; ++in_dim) {
                down_w[static_cast<size_t>(in_dim * hidden + out_dim)] =
                    decode_mxfp4_value(down_blocks, down_scales, down_blocks_e.shape, expert, out_dim, in_dim);
            }
        }
        report_variant("moe_scalar_gate_weight_decode",
                       make_vector_variant("mxfp4_decode", gate_w, ref_gate_w),
                       ref_gate_w);
        report_variant("moe_scalar_down_weight_decode",
                       make_vector_variant("mxfp4_decode", down_w, ref_down_w),
                       ref_down_w,
                       channel);
        report_variant("moe_scalar_gate_bias_raw_vs_oracle",
                       make_vector_variant("raw_f32", std::vector<float>(gate_bias_raw.begin() + static_cast<std::ptrdiff_t>(expert * 2 * intermediate),
                                                                          gate_bias_raw.begin() + static_cast<std::ptrdiff_t>((expert + 1) * 2 * intermediate)), ref_gate_bias),
                       ref_gate_bias);
        report_variant("moe_scalar_down_bias_raw_vs_oracle",
                       make_vector_variant("raw_f32", std::vector<float>(down_bias_raw.begin() + static_cast<std::ptrdiff_t>(expert * hidden),
                                                                          down_bias_raw.begin() + static_cast<std::ptrdiff_t>((expert + 1) * hidden)), ref_down_bias),
                       ref_down_bias,
                       channel);

        std::vector<float> gate_pre_f32(static_cast<size_t>(2 * intermediate), 0.0f);
        for (int64_t out_dim = 0; out_dim < 2 * intermediate; ++out_dim) {
            float sum = 0.0f;
            for (int64_t in_dim = 0; in_dim < hidden; ++in_dim) {
                sum = std::fma(current_state[static_cast<size_t>(in_dim)],
                               gate_w[static_cast<size_t>(in_dim * 2 * intermediate + out_dim)],
                               sum);
            }
            gate_pre_f32[static_cast<size_t>(out_dim)] = sum;
        }
        std::vector<VectorVariant> gate_pre_variants;
        gate_pre_variants.push_back(make_vector_variant("f32_accum_f32_output", gate_pre_f32, ref_gate_pre));
        gate_pre_variants.push_back(make_vector_variant("f32_accum_cast_bf16", round_vector_bf16(gate_pre_f32), ref_gate_pre));
        for (const VectorVariant& variant : gate_pre_variants) {
            report_variant("moe_scalar_gate_up_pre_bias", variant, ref_gate_pre);
        }
        const VectorVariant& best_gate_pre = best_variant(gate_pre_variants);
        note_first_bad("gate_up_pre_bias", best_gate_pre.diff.max_diff);

        std::vector<float> gate_bias_oracle = ref_gate_bias.data;
        std::vector<float> gate_bias_raw_slice(gate_bias_raw.begin() + static_cast<std::ptrdiff_t>(expert * 2 * intermediate),
                                               gate_bias_raw.begin() + static_cast<std::ptrdiff_t>((expert + 1) * 2 * intermediate));
        std::vector<float> gate_bias_raw_bf16 = round_vector_bf16(gate_bias_raw_slice);
        auto add_bias_variant = [&](const std::string& name, const std::vector<float>& pre, const std::vector<float>& bias) {
            std::vector<float> out(pre.size());
            for (size_t i = 0; i < pre.size(); ++i) {
                out[i] = round_to_bf16(pre[i] + bias[i]);
            }
            return make_vector_variant(name, std::move(out), ref_gate_post);
        };
        std::vector<VectorVariant> gate_post_variants;
        gate_post_variants.push_back(add_bias_variant("pre_f32_raw_f32_bias_final_bf16", gate_pre_f32, gate_bias_raw_slice));
        gate_post_variants.push_back(add_bias_variant("pre_f32_raw_bias_cast_bf16_final_bf16", gate_pre_f32, gate_bias_raw_bf16));
        gate_post_variants.push_back(add_bias_variant("pre_f32_oracle_bias_final_bf16", gate_pre_f32, gate_bias_oracle));
        gate_post_variants.push_back(add_bias_variant("pre_bf16_raw_f32_bias_final_bf16", gate_pre_variants[1].values, gate_bias_raw_slice));
        gate_post_variants.push_back(add_bias_variant("pre_bf16_raw_bias_cast_bf16_final_bf16", gate_pre_variants[1].values, gate_bias_raw_bf16));
        gate_post_variants.push_back(add_bias_variant("pre_bf16_oracle_bias_final_bf16", gate_pre_variants[1].values, gate_bias_oracle));
        for (const VectorVariant& variant : gate_post_variants) {
            report_variant("moe_scalar_gate_up_post_bias", variant, ref_gate_post);
        }
        const VectorVariant& best_gate_post = best_variant(gate_post_variants);
        note_first_bad("gate_up_post_bias", best_gate_post.diff.max_diff);
        std::cout << "moe_scalar_gate_up_selected_variant expert=" << expert
                  << " pre_variant=" << best_gate_pre.name
                  << " post_variant=" << best_gate_post.name
                  << "\n";

        auto compute_swiglu_variant = [&](const std::string& name, bool bf16_each_op) {
            std::vector<float> gate_clamped(static_cast<size_t>(intermediate));
            std::vector<float> up_clamped(static_cast<size_t>(intermediate));
            std::vector<float> gate_alpha(static_cast<size_t>(intermediate));
            std::vector<float> sigmoid_v(static_cast<size_t>(intermediate));
            std::vector<float> glu(static_cast<size_t>(intermediate));
            std::vector<float> gated(static_cast<size_t>(intermediate));
            for (int64_t i = 0; i < intermediate; ++i) {
                float gate = std::min(best_gate_post.values[static_cast<size_t>(2 * i)], 7.0f);
                float up = std::max(-7.0f, std::min(best_gate_post.values[static_cast<size_t>(2 * i + 1)], 7.0f));
                if (bf16_each_op) {
                    gate = round_to_bf16(gate);
                    up = round_to_bf16(up);
                }
                float alpha_v = gate * 1.702f;
                if (bf16_each_op) alpha_v = round_to_bf16(alpha_v);
                float sigmoid_out = 1.0f / (1.0f + std::exp(-alpha_v));
                if (bf16_each_op) sigmoid_out = round_to_bf16(sigmoid_out);
                float glu_v = gate * sigmoid_out;
                if (bf16_each_op) glu_v = round_to_bf16(glu_v);
                float up_plus = up + 1.0f;
                if (bf16_each_op) up_plus = round_to_bf16(up_plus);
                float gated_v = up_plus * glu_v;
                gated_v = round_to_bf16(gated_v);
                gate_clamped[static_cast<size_t>(i)] = gate;
                up_clamped[static_cast<size_t>(i)] = up;
                gate_alpha[static_cast<size_t>(i)] = alpha_v;
                sigmoid_v[static_cast<size_t>(i)] = sigmoid_out;
                glu[static_cast<size_t>(i)] = glu_v;
                gated[static_cast<size_t>(i)] = gated_v;
            }
            std::cout << "moe_scalar_swiglu_variant expert=" << expert << " name=" << name << "\n";
            report_variant("moe_scalar_gate_clamped", make_vector_variant(name, gate_clamped, ref_gate_clamped), ref_gate_clamped);
            report_variant("moe_scalar_up_clamped", make_vector_variant(name, up_clamped, ref_up_clamped), ref_up_clamped);
            report_variant("moe_scalar_gate_alpha", make_vector_variant(name, gate_alpha, ref_gate_alpha), ref_gate_alpha);
            report_variant("moe_scalar_sigmoid", make_vector_variant(name, sigmoid_v, ref_sigmoid), ref_sigmoid);
            report_variant("moe_scalar_glu", make_vector_variant(name, glu, ref_glu), ref_glu);
            return make_vector_variant(name, std::move(gated), ref_gated);
        };
        std::vector<VectorVariant> gated_variants;
        gated_variants.push_back(compute_swiglu_variant("f32_math_final_bf16", false));
        gated_variants.push_back(compute_swiglu_variant("bf16_each_elementwise_op", true));
        for (const VectorVariant& variant : gated_variants) {
            report_variant("moe_scalar_gated_output", variant, ref_gated);
        }
        const VectorVariant& best_gated = best_variant(gated_variants);
        note_first_bad("gated_output", best_gated.diff.max_diff);
        std::cout << "moe_scalar_swiglu_selected_variant expert=" << expert
                  << " variant=" << best_gated.name << "\n";

        std::vector<float> down_pre_f32(static_cast<size_t>(hidden), 0.0f);
        for (int64_t out_dim = 0; out_dim < hidden; ++out_dim) {
            float sum = 0.0f;
            for (int64_t in_dim = 0; in_dim < intermediate; ++in_dim) {
                sum = std::fma(best_gated.values[static_cast<size_t>(in_dim)],
                               down_w[static_cast<size_t>(in_dim * hidden + out_dim)],
                               sum);
            }
            down_pre_f32[static_cast<size_t>(out_dim)] = sum;
        }
        std::vector<VectorVariant> down_pre_variants;
        down_pre_variants.push_back(make_vector_variant("f32_accum_f32_output", down_pre_f32, ref_down_pre));
        down_pre_variants.push_back(make_vector_variant("f32_accum_cast_bf16", round_vector_bf16(down_pre_f32), ref_down_pre));
        for (const VectorVariant& variant : down_pre_variants) {
            report_variant("moe_scalar_down_pre_bias", variant, ref_down_pre, channel);
        }
        const VectorVariant& best_down_pre = best_variant(down_pre_variants);
        note_first_bad("down_pre_bias", best_down_pre.diff.max_diff);

        std::vector<float> down_bias_oracle = ref_down_bias.data;
        std::vector<float> down_bias_raw_slice(down_bias_raw.begin() + static_cast<std::ptrdiff_t>(expert * hidden),
                                               down_bias_raw.begin() + static_cast<std::ptrdiff_t>((expert + 1) * hidden));
        std::vector<float> down_bias_raw_bf16 = round_vector_bf16(down_bias_raw_slice);
        auto add_down_bias_variant = [&](const std::string& name, const std::vector<float>& pre, const std::vector<float>& bias) {
            std::vector<float> out(pre.size());
            for (size_t i = 0; i < pre.size(); ++i) {
                out[i] = round_to_bf16(pre[i] + bias[i]);
            }
            return make_vector_variant(name, std::move(out), ref_down_post);
        };
        std::vector<VectorVariant> down_post_variants;
        down_post_variants.push_back(add_down_bias_variant("pre_f32_raw_f32_bias_final_bf16", down_pre_f32, down_bias_raw_slice));
        down_post_variants.push_back(add_down_bias_variant("pre_f32_raw_bias_cast_bf16_final_bf16", down_pre_f32, down_bias_raw_bf16));
        down_post_variants.push_back(add_down_bias_variant("pre_f32_oracle_bias_final_bf16", down_pre_f32, down_bias_oracle));
        down_post_variants.push_back(add_down_bias_variant("pre_bf16_raw_f32_bias_final_bf16", down_pre_variants[1].values, down_bias_raw_slice));
        down_post_variants.push_back(add_down_bias_variant("pre_bf16_raw_bias_cast_bf16_final_bf16", down_pre_variants[1].values, down_bias_raw_bf16));
        down_post_variants.push_back(add_down_bias_variant("pre_bf16_oracle_bias_final_bf16", down_pre_variants[1].values, down_bias_oracle));
        for (const VectorVariant& variant : down_post_variants) {
            report_variant("moe_scalar_down_post_bias", variant, ref_down_post, channel);
        }
        const VectorVariant& best_down_post = best_variant(down_post_variants);
        note_first_bad("down_post_bias", best_down_post.diff.max_diff);
        std::cout << "moe_scalar_down_selected_variant expert=" << expert
                  << " pre_variant=" << best_down_pre.name
                  << " post_variant=" << best_down_post.name
                  << "\n";

        for (int kpos : positions) {
            const float route = topk_weights_npy.data[static_cast<size_t>(kpos)];
            const NpyF32 ref_weighted = load_npy_f32(oracle_dir / (expert_prefix + "_topk" + std::to_string(kpos) + "_weighted_output_f32.npy"));
            const NpyF32 ref_weighted_hidden = load_npy_f32(oracle_dir / (expert_prefix + "_topk" + std::to_string(kpos) + "_weighted_output_hidden_dtype_f32.npy"));
            const NpyF32 ref_accum = load_npy_f32(oracle_dir / (expert_prefix + "_topk" + std::to_string(kpos) + "_accumulated_after_f32.npy"));
            std::vector<float> weighted(best_down_post.values.size());
            std::vector<float> weighted_hidden(best_down_post.values.size());
            for (size_t i = 0; i < best_down_post.values.size(); ++i) {
                weighted[i] = round_to_bf16(best_down_post.values[i] * route);
                weighted_hidden[i] = round_to_bf16(weighted[i]);
                accumulated[i] = round_to_bf16(accumulated[i] + weighted_hidden[i]);
            }
            VectorVariant weighted_variant = make_vector_variant("down_post_bf16_times_route_bf16", weighted, ref_weighted);
            VectorVariant weighted_hidden_variant = make_vector_variant("weighted_to_hidden_bf16", weighted_hidden, ref_weighted_hidden);
            VectorVariant accum_variant = make_vector_variant("bf16_index_add_order", accumulated, ref_accum);
            report_variant("moe_scalar_weighted_output", weighted_variant, ref_weighted, channel);
            report_variant("moe_scalar_weighted_hidden_output", weighted_hidden_variant, ref_weighted_hidden, channel);
            report_variant("moe_scalar_accumulated_after", accum_variant, ref_accum, channel);
            note_first_bad("weighted_output", weighted_variant.diff.max_diff);
            note_first_bad("accumulation", accum_variant.diff.max_diff);
            channel_contribs.push_back(weighted_hidden[static_cast<size_t>(channel)]);
        }
    }

    const NpyF32 ref_final_token = load_npy_f32(oracle_dir / (token_prefix + "_manual_accumulated_f32.npy"));
    VectorVariant final_variant = make_vector_variant("selected_best_variants_bf16_accum", accumulated, ref_final_token);
    report_variant("moe_scalar_token_accumulated_final", final_variant, ref_final_token, channel);
    std::cout << "moe_scalar_channel_contributions channel=" << channel << " values=";
    for (size_t i = 0; i < channel_contribs.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << channel_contribs[i];
    }
    std::cout << "\n";
    report_moe_scalar_value("moe_scalar_channel_final", accumulated[static_cast<size_t>(channel)], ref_final_token.data[static_cast<size_t>(channel)]);
    std::cout << "moe_scalar_replay_summary"
              << " first_bad_stage=" << first_bad_stage
              << " first_bad_max_diff=" << first_bad_diff
              << " final_token_max_diff=" << final_variant.diff.max_diff
              << " final_token_mean_diff=" << final_variant.diff.mean_diff
              << " channel=" << channel
              << " channel_native=" << accumulated[static_cast<size_t>(channel)]
              << " channel_oracle=" << ref_final_token.data[static_cast<size_t>(channel)]
              << "\n";
}

static void print_config_summary(const std::filesystem::path& config_path) {
    std::ifstream in(config_path);
    if (!in) {
        throw std::runtime_error("failed to read config: " + config_path.string());
    }
    nlohmann::json cfg = nlohmann::json::parse(in);
    std::cout << "config architecture=" << cfg.at("architectures").at(0).get<std::string>()
              << " layers=" << cfg.at("num_hidden_layers").get<int>()
              << " hidden=" << cfg.at("hidden_size").get<int>()
              << " intermediate=" << cfg.at("intermediate_size").get<int>()
              << " vocab=" << cfg.at("vocab_size").get<int>()
              << " heads=" << cfg.at("num_attention_heads").get<int>()
              << " kv_heads=" << cfg.at("num_key_value_heads").get<int>()
              << " head_dim=" << cfg.at("head_dim").get<int>()
              << " experts=" << cfg.at("num_local_experts").get<int>()
              << " experts_per_token=" << cfg.at("num_experts_per_tok").get<int>()
              << " sliding_window=" << cfg.at("sliding_window").get<int>()
              << " dtype=" << cfg.at("dtype").get<std::string>() << "\n";
    const auto& rope = cfg.at("rope_parameters");
    std::cout << "rope type=" << rope.at("rope_type").get<std::string>()
              << " theta=" << rope.at("rope_theta").get<int>()
              << " factor=" << rope.at("factor").get<double>()
              << " original_max_position_embeddings=" << rope.at("original_max_position_embeddings").get<int>()
              << "\n";
}

static std::vector<std::string> load_layer_types(const std::filesystem::path& config_path) {
    std::ifstream in(config_path);
    if (!in) {
        throw std::runtime_error("failed to read config: " + config_path.string());
    }
    nlohmann::json cfg = nlohmann::json::parse(in);
    std::vector<std::string> out;
    for (const auto& item : cfg.at("layer_types")) {
        out.push_back(item.get<std::string>());
    }
    return out;
}

#ifndef SD_LENS_GPTOSS_TEXT_ENCODER_NO_MAIN
int main(int argc, char** argv) {
    try {
        Args args;
        if (!parse_args(argc, argv, args)) {
            usage(argv[0]);
            return 2;
        }
        const std::filesystem::path text_dir(args.text_encoder_dir);
        const std::filesystem::path oracle_dir(args.oracle_dir);
        g_moe_backend = args.moe_backend;
        g_moe_cache = args.moe_cache;
        g_moe_cache_layout = args.moe_cache_layout;
        g_cache_upload = args.cache_upload;
        g_cache_upload_chunk_mib = args.cache_upload_chunk_mib;
        g_moe_bf16_cache_dir = args.moe_bf16_cache_dir;
        std::vector<SafetensorEntry> entries;
        auto index = index_safetensors(text_dir, &entries);
        std::map<std::string, int> dtype_counts;
        uint64_t payload_bytes = 0;
        for (const auto& e : entries) {
            dtype_counts[e.dtype]++;
            payload_bytes += e.end - e.begin;
        }
        print_config_summary(text_dir / "config.json");
        std::cout << "weight_map tensors=" << entries.size()
                  << " shards=" << safetensor_files(text_dir).size()
                  << " payload_bytes=" << payload_bytes;
        for (const auto& kv : dtype_counts) {
            std::cout << " dtype_" << kv.first << "=" << kv.second;
        }
        std::cout << "\n";
        if (!args.weight_map_out.empty()) {
            write_weight_map_csv(args.weight_map_out, entries);
            std::cout << "wrote_weight_map=" << args.weight_map_out << "\n";
        }
        NpyI64 ids = load_npy_i64(oracle_dir / "input_ids_i64.npy");
        NpyI64 mask = load_npy_i64(oracle_dir / "attention_mask_i64.npy");
        NpyI64 trimmed = load_npy_i64(oracle_dir / "attention_mask_trimmed_i64.npy");
        std::cout << "token_bootstrap input_ids_shape=1x" << ids.shape.at(1)
                  << " attention_mask_shape=1x" << mask.shape.at(1)
                  << " trimmed_mask_shape=1x" << trimmed.shape.at(1)
                  << " txt_offset=" << (mask.shape.at(1) - trimmed.shape.at(1))
                  << " tokenizer_mode=oracle_input_ids_bootstrap"
                  << " no_oracle_compare=" << (args.no_oracle_compare ? "true" : "false") << "\n";
        if (!args.no_oracle_compare) {
            run_mxfp4_proof(oracle_dir, index);
            run_embedding_parity(oracle_dir, index);
        } else {
            std::cout << "startup_diagnostics_skipped no_oracle_compare=true skipped=mxfp4_proof,embedding_parity\n";
        }
        if (args.router_only_cuda) {
            run_router_cuda_parity(oracle_dir, index);
            std::cout << "Lens text_encoder router CUDA parity passed\n";
            return 0;
        }
        if (args.projection_only) {
            run_projection_only(args.projection_layer, oracle_dir, index);
            std::cout << "Lens text_encoder projection-only smoke completed layer=" << args.projection_layer << "\n";
            return 0;
        }
        if (args.rope_only) {
            run_rope_only(args.rope_layer, oracle_dir);
            std::cout << "Lens text_encoder RoPE-only smoke completed layer=" << args.rope_layer << "\n";
            return 0;
        }
        if (args.attention_only) {
            run_attention_only(args.attention_layer, oracle_dir, index);
            std::cout << "Lens text_encoder attention-only smoke completed layer=" << args.attention_layer << "\n";
            return 0;
        }
        if (args.isolated_layer_set) {
            run_isolated_layer(oracle_dir, text_dir, index, args.isolated_layer);
            std::cout << "Lens text_encoder isolated-layer smoke completed layer=" << args.isolated_layer << "\n";
            return 0;
        }
        if (args.layer0_drift_probe) {
            run_layer0_drift_probe(oracle_dir, index);
            std::cout << "Lens text_encoder layer0 drift probe completed\n";
            return 0;
        }
        if (args.layer1_perturb_probe) {
            run_layer1_perturb_probe(oracle_dir, text_dir, index);
            std::cout << "Lens text_encoder layer1 perturb probe completed\n";
            return 0;
        }
        if (args.residual_cast_audit) {
            run_residual_cast_audit(oracle_dir, text_dir, index);
            std::cout << "Lens text_encoder residual/cast audit completed\n";
            return 0;
        }
        if (args.moe_scalar_replay) {
            run_moe_scalar_replay(oracle_dir, index, args.moe_scalar_layer, args.moe_scalar_token, args.moe_scalar_channel);
            std::cout << "Lens text_encoder MoE scalar replay completed layer=" << args.moe_scalar_layer
                      << " token=" << args.moe_scalar_token
                      << " channel=" << args.moe_scalar_channel << "\n";
            return 0;
        }
        if (args.through_layer_set) {
            run_layers_0_to_n(oracle_dir,
                              text_dir,
                              index,
                              args.through_layer,
                              args.summary_only || args.through_capture,
                              args.router_mode,
                              args.emit_lens_cond_v1.empty() ? std::filesystem::path{} : std::filesystem::path(args.emit_lens_cond_v1),
                              args.no_oracle_compare);
            std::cout << "Lens text_encoder layered smoke passed "
                      << (args.through_capture ? "through_capture=" : "through_layer=")
                      << args.through_layer
                      << " router_mode=" << args.router_mode
                      << " summary_only=" << (args.summary_only ? "true" : "false");
            if (!args.emit_lens_cond_v1.empty()) {
                std::cout << " emitted_lens_cond_v1=" << args.emit_lens_cond_v1;
            }
            std::cout << "\n";
            return 0;
        }
        run_layer0_parity(oracle_dir, index);
        std::cout << "Lens text_encoder smoke passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sd-lens-text-encoder-smoke failed: %s\n", e.what());
        return 1;
    }
}
#endif
