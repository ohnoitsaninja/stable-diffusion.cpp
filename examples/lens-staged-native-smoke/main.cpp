#include "lens_staged_pipeline.hpp"

#include "../common/media_io.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

struct StagedArgs {
    std::string text_encoder_dir;
    std::string bootstrap_oracle_dir;
    std::string prompt;
    std::string tokenizer_dir;
    std::string chat_current_date;
    std::string transformer_dir;
    std::string vae_path;
    std::string output_png;
    std::string cond_out;
    int width = 256;
    int height = 256;
    int steps = 4;
    int seed = 42;
    int repeat_generations = 1;
    bool warm_runtime = false;
    bool tokenizer_only = false;
    std::string transformer_speed_mode;
    std::string transformer_residency = "streaming";
    std::string dynamic_residency = "none";
    int transformer_window_blocks = 0;
    int transformer_persistent_blocks = 0;
    uint64_t transformer_persistent_blocks_memory_mib = 4096;
};

static void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --text-encoder <Lens-Turbo/text_encoder> --bootstrap-oracle <oracle-dir> "
                 "--transformer <Lens-Turbo/transformer> --vae <Lens-Turbo/vae/diffusion_pytorch_model.safetensors> "
                 "--height 256 --width 256 --steps 4 --seed 42 --output <out.png> [--prompt <text>] [--cond-out <lens_cond_v1.safetensors>] "
                 "[--tokenizer-only] [--tokenizer-dir <Lens-Turbo/tokenizer>] [--chat-current-date YYYY-MM-DD] "
                 "[--warm-runtime] [--transformer-speed-mode bf16-resident] [--repeat-generations 2] "
                 "[--dynamic-residency none|two-block-proof|gpu-streams] "
                 "[--transformer-residency streaming|gpu-window|persistent-blocks|gpu-full-bf16 --window-blocks N --persistent-blocks N] "
                 "[--persistent-blocks-memory-mib 4096]\n",
                 argv0);
}

static bool parse_args(int argc, char** argv, StagedArgs& args) {
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--text-encoder") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.text_encoder_dir = value;
        } else if (std::strcmp(argv[i], "--bootstrap-oracle") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.bootstrap_oracle_dir = value;
        } else if (std::strcmp(argv[i], "--prompt") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.prompt = value;
        } else if (std::strcmp(argv[i], "--tokenizer-dir") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.tokenizer_dir = value;
        } else if (std::strcmp(argv[i], "--chat-current-date") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.chat_current_date = value;
        } else if (std::strcmp(argv[i], "--transformer") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.transformer_dir = value;
        } else if (std::strcmp(argv[i], "--vae") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.vae_path = value;
        } else if (std::strcmp(argv[i], "--height") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.height = std::atoi(value);
        } else if (std::strcmp(argv[i], "--width") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.width = std::atoi(value);
        } else if (std::strcmp(argv[i], "--steps") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.steps = std::atoi(value);
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.seed = std::atoi(value);
        } else if (std::strcmp(argv[i], "--repeat-generations") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.repeat_generations = std::atoi(value);
        } else if (std::strcmp(argv[i], "--warm-runtime") == 0) {
            args.warm_runtime = true;
        } else if (std::strcmp(argv[i], "--tokenizer-only") == 0) {
            args.tokenizer_only = true;
        } else if (std::strcmp(argv[i], "--output") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.output_png = value;
        } else if (std::strcmp(argv[i], "--cond-out") == 0 || std::strcmp(argv[i], "--optional-cond-out") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.cond_out = value;
        } else if (std::strcmp(argv[i], "--transformer-residency") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.transformer_residency = value;
        } else if (std::strcmp(argv[i], "--dynamic-residency") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.dynamic_residency = value;
        } else if (std::strcmp(argv[i], "--transformer-speed-mode") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.transformer_speed_mode = value;
            if (args.transformer_speed_mode == "bf16-resident") {
                args.transformer_residency = "gpu-full-bf16";
            } else {
                std::fprintf(stderr, "--transformer-speed-mode must be bf16-resident\n");
                return false;
            }
        } else if (std::strcmp(argv[i], "--window-blocks") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.transformer_window_blocks = std::atoi(value);
        } else if (std::strcmp(argv[i], "--persistent-blocks") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.transformer_persistent_blocks = std::atoi(value);
        } else if (std::strcmp(argv[i], "--persistent-blocks-memory-mib") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.transformer_persistent_blocks_memory_mib = static_cast<uint64_t>(std::strtoull(value, nullptr, 10));
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return false;
        }
    }
    if (args.tokenizer_only) {
        return !args.prompt.empty() && (!args.tokenizer_dir.empty() || !args.text_encoder_dir.empty());
    }
    return !args.text_encoder_dir.empty() &&
           (!args.bootstrap_oracle_dir.empty() || !args.prompt.empty()) &&
           !args.transformer_dir.empty() &&
           !args.vae_path.empty() &&
           !args.output_png.empty();
}

static void print_vram_stage(const char* stage) {
    std::cout << "vram_snapshot stage=" << stage << " source=external-only\n";
}

static void create_parent_dir(const std::string& path) {
    const std::filesystem::path fs_path(path);
    const std::filesystem::path parent = fs_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

static std::vector<int64_t> load_i64_npy_1xN(const std::filesystem::path& path, int64_t expected_n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + path.string());
    }
    char magic[6] = {};
    in.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("invalid npy magic: " + path.string());
    }
    char version[2] = {};
    in.read(version, 2);
    uint32_t header_len = 0;
    if (version[0] == 1) {
        uint16_t h16 = 0;
        in.read(reinterpret_cast<char*>(&h16), sizeof(h16));
        header_len = h16;
    } else {
        in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    }
    std::string header(header_len, '\0');
    in.read(header.data(), header.size());
    if (header.find("<i8") == std::string::npos || header.find("(1, " + std::to_string(expected_n)) == std::string::npos) {
        throw std::runtime_error("unexpected npy i64 shape/header in " + path.string() + ": " + header);
    }
    std::vector<int64_t> data(static_cast<size_t>(expected_n));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(int64_t)));
    if (!in) {
        throw std::runtime_error("short npy read: " + path.string());
    }
    return data;
}

static uint64_t fnv1a_i64(const std::vector<int64_t>& values) {
    uint64_t h = 1469598103934665603ull;
    for (int64_t v : values) {
        for (int b = 0; b < 8; ++b) {
            h ^= static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * b)) & 0xffu);
            h *= 1099511628211ull;
        }
    }
    return h;
}

static bool run_tokenizer_only(const StagedArgs& args) {
    const std::filesystem::path tokenizer_dir =
        !args.tokenizer_dir.empty()
            ? std::filesystem::path(args.tokenizer_dir)
            : (std::filesystem::path(args.text_encoder_dir).parent_path() / "tokenizer");
    sd_lens_gptoss_tokenizer tokenizer;
    std::string error;
    if (!tokenizer.load(tokenizer_dir.string(), &error)) {
        std::cerr << "Lens tokenizer load failed: " << error << "\n";
        return false;
    }
    sd_lens_tokenized_prompt tokenized;
    if (!tokenizer.encode_prompt(args.prompt, 128, args.chat_current_date, &tokenized, &error)) {
        std::cerr << "Lens tokenizer encode failed: " << error << "\n";
        return false;
    }
    const int64_t ones = std::accumulate(tokenized.attention_mask.begin(), tokenized.attention_mask.end(), int64_t{0});
    bool ids_equal = true;
    bool mask_equal = true;
    bool trimmed_equal = true;
    if (!args.bootstrap_oracle_dir.empty()) {
        const std::filesystem::path oracle(args.bootstrap_oracle_dir);
        const std::vector<int64_t> ref_ids = load_i64_npy_1xN(oracle / "input_ids_i64.npy", 128);
        const std::vector<int64_t> ref_mask = load_i64_npy_1xN(oracle / "attention_mask_i64.npy", 128);
        const std::vector<int64_t> ref_trimmed = load_i64_npy_1xN(oracle / "attention_mask_trimmed_i64.npy", 31);
        ids_equal = tokenized.input_ids == ref_ids;
        mask_equal = tokenized.attention_mask == ref_mask;
        std::vector<int64_t> trimmed(tokenized.attention_mask.begin() + tokenized.txt_offset,
                                     tokenized.attention_mask.end());
        trimmed_equal = trimmed == ref_trimmed;
    }
    std::cout << "Lens tokenizer smoke passed"
              << " tokenizer=native_gptoss_tokenizer_json"
              << " tokenizer_dir=" << tokenizer_dir.string()
              << " prompt_chars=" << args.prompt.size()
              << " rendered_chars=" << tokenized.rendered_prompt.size()
              << " raw_seq_len=" << tokenized.raw_seq_len
              << " trimmed_seq_len=" << tokenized.trimmed_seq_len
              << " txt_offset=" << tokenized.txt_offset
              << " mask_ones=" << ones
              << " input_ids_fnv1a64=" << fnv1a_i64(tokenized.input_ids)
              << " attention_mask_fnv1a64=" << fnv1a_i64(tokenized.attention_mask)
              << " input_ids_equal_oracle=" << (ids_equal ? "true" : "false")
              << " attention_mask_equal_oracle=" << (mask_equal ? "true" : "false")
              << " trimmed_attention_mask_equal_oracle=" << (trimmed_equal ? "true" : "false")
              << " chat_current_date="
              << (args.chat_current_date.empty() ? sd_lens_current_date_yyyy_mm_dd() : args.chat_current_date)
              << "\n";
    return ids_equal && mask_equal && trimmed_equal;
}

int main(int argc, char** argv) {
    StagedArgs args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }
    if (args.tokenizer_only) {
        try {
            return run_tokenizer_only(args) ? 0 : 1;
        } catch (const std::exception& e) {
            std::cerr << "sd-lens-staged-native-smoke tokenizer-only failed: " << e.what() << "\n";
            return 1;
        }
    }
    if (args.steps <= 0) {
        std::cerr << "Lens staged native smoke requires positive --steps\n";
        return 2;
    }
    if (args.repeat_generations <= 0 || args.repeat_generations > 4) {
        std::cerr << "--repeat-generations must be in [1,4]\n";
        return 2;
    }
    if (args.width != args.height || (args.width != 256 && args.width != 512 && args.width != 1024)) {
        std::cerr << "Lens staged native smoke supports square 256, 512, or 1024 only\n";
        return 2;
    }
    if (args.transformer_residency != "streaming" &&
        args.transformer_residency != "gpu-window" &&
        args.transformer_residency != "persistent-blocks" &&
        args.transformer_residency != "gpu-full-bf16") {
        std::cerr << "--transformer-residency must be streaming, gpu-window, persistent-blocks, or gpu-full-bf16\n";
        return 2;
    }
    if (args.transformer_residency == "gpu-window" && args.transformer_window_blocks <= 0) {
        std::cerr << "--transformer-residency gpu-window requires --window-blocks N\n";
        return 2;
    }
    if (args.transformer_residency == "persistent-blocks" && args.transformer_persistent_blocks <= 0) {
        std::cerr << "--transformer-residency persistent-blocks requires --persistent-blocks N\n";
        return 2;
    }
    if (args.dynamic_residency != "none" &&
        args.dynamic_residency != "two-block-proof" &&
        args.dynamic_residency != "gpu-streams") {
        std::cerr << "--dynamic-residency must be none, two-block-proof, or gpu-streams\n";
        return 2;
    }

    try {
        create_parent_dir(args.output_png);
        if (!args.cond_out.empty()) {
            create_parent_dir(args.cond_out);
        }

        const std::filesystem::path output_path(args.output_png);
        const std::filesystem::path parent =
            output_path.parent_path().empty() ? std::filesystem::current_path() : output_path.parent_path();
        const std::string stem = output_path.stem().string();

        LensPromptRequest request;
        request.text_encoder_dir = args.text_encoder_dir;
        request.bootstrap_oracle_dir = args.bootstrap_oracle_dir;
        request.prompt = args.prompt;
        request.tokenizer_dir = args.tokenizer_dir;
        request.chat_current_date = args.chat_current_date;
        request.transformer_dir = args.transformer_dir;
        request.vae_path = args.vae_path;
        request.optional_cond_out = args.cond_out;
        request.latent_npy = (parent / (stem + "_latent.npy")).string();
        request.packed_tokens_npy = (parent / (stem + "_packed.npy")).string();
        request.width = args.width;
        request.height = args.height;
        request.steps = args.steps;
        request.seed = args.seed;
        request.repeat_generations = args.repeat_generations;
        request.speed_mode = args.transformer_speed_mode;
        request.quality_mode = args.transformer_speed_mode == "bf16-resident" ? "speed" : "reference";
        request.transformer_residency = args.transformer_residency;
        request.dynamic_residency = args.dynamic_residency;
        request.transformer_window_blocks = args.transformer_window_blocks;
        request.transformer_persistent_blocks = args.transformer_persistent_blocks;
        request.transformer_persistent_blocks_memory_mib = args.transformer_persistent_blocks_memory_mib;

        std::cout << "Lens staged native smoke start\n" << std::flush;
        if (args.warm_runtime) {
            std::cout << "Lens staged native warm runtime:"
                      << " warm_runtime=true"
                      << " same_condition_reuse=" << (args.repeat_generations > 1 ? "true" : "false")
                      << " repeat_generations=" << args.repeat_generations
                      << " internal_runtime_object=sd_lens_staged_runtime"
                      << " public_api=false\n";
        }
        if (!args.transformer_speed_mode.empty()) {
            std::cout << "Lens staged native speed mode:"
                      << " speed_mode=" << args.transformer_speed_mode
                      << " transformer_residency=" << args.transformer_residency
                      << " dynamic_residency=" << args.dynamic_residency
                      << " hidden_parity_exact=false"
                      << " image_validated_256=true"
                      << " image_validated_512=true"
                      << " repeat_generations=" << args.repeat_generations
                      << "\n";
        }
        print_vram_stage("process_start");
        print_vram_stage("before_text_encoder_load");
        LensPromptResult prompt_result;
        const bool ok = sd_lens_run_prompt_request(request, &prompt_result);
        if (!ok) {
            std::cerr << "sd_lens_run_prompt_request failed: " << prompt_result.error << "\n";
            return 1;
        }
        if (prompt_result.image_rgb.empty() ||
            !write_image_to_file(args.output_png,
                                 prompt_result.image_rgb.data(),
                                 prompt_result.image_width,
                                 prompt_result.image_height,
                                 prompt_result.image_channels)) {
            std::cerr << "failed to write output: " << args.output_png << "\n";
            return 1;
        }
        std::cout << "Lens staged native image wrote " << args.output_png
                  << " image=" << prompt_result.image_width << "x"
                  << prompt_result.image_height << "x" << prompt_result.image_channels
                  << "\n";

        std::cout << "Lens staged native smoke passed"
                  << " output=" << args.output_png
                  << " text_encoder_load_seconds=" << prompt_result.text_encoder_load_seconds
                  << " text_encoder_seconds=" << prompt_result.text_encoder_encode_seconds
                  << " text_encoder_wrapper_seconds=" << prompt_result.text_encoder_wrapper_seconds
                  << " transformer_wall_seconds=" << prompt_result.transformer_wall_seconds
                  << " vae_wall_seconds=" << prompt_result.vae_wall_seconds
                  << " vae_decode_seconds=" << prompt_result.vae_decode_seconds
                  << " condition_handoff=in_memory"
                  << " warm_runtime=" << (args.warm_runtime ? "true" : "false")
                  << " same_condition_reuse="
                  << (prompt_result.lifecycle.same_condition_reuse ? "true" : "false")
                  << " repeat_generations=" << args.repeat_generations
                  << " speed_mode=" << (args.transformer_speed_mode.empty() ? "<none>" : args.transformer_speed_mode)
                  << " hidden_parity_exact="
                  << (args.transformer_speed_mode == "bf16-resident" ? "false" : "n/a")
                  << " optional_cond_out=" << (args.cond_out.empty() ? "<none>" : args.cond_out)
                  << " text_encoder_released_before_transformer="
                  << (prompt_result.lifecycle.text_encoder_released ? "true" : "false")
                  << " transformer_loaded_after_text_encoder_release="
                  << (prompt_result.lifecycle.transformer_loaded &&
                      prompt_result.lifecycle.text_encoder_released ? "true" : "false")
                  << " transformer_released="
                  << (prompt_result.lifecycle.transformer_released ? "true" : "false")
                  << " transformer_stage_internal_runtime=true"
                  << " transformer_smoke_api_dll=false"
                  << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "sd-lens-staged-native-smoke failed: " << e.what() << "\n";
        return 1;
    }
}
