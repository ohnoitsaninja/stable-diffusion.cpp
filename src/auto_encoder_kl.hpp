#ifndef __AUTO_ENCODER_KL_HPP__
#define __AUTO_ENCODER_KL_HPP__

#include "vae.hpp"

#include <cstdlib>
#include <limits>
#include <utility>

/*================================================== AutoEncoderKL ===================================================*/

#define VAE_GRAPH_SIZE 20480

class ResnetBlock : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;

public:
    ResnetBlock(int64_t in_channels,
                int64_t out_channels)
        : in_channels(in_channels),
          out_channels(out_channels) {
        // temb_channels is always 0
        blocks["norm1"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(in_channels));
        blocks["conv1"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));

        blocks["norm2"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(out_channels));
        blocks["conv2"] = std::shared_ptr<GGMLBlock>(new Conv2d(out_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));

        if (out_channels != in_channels) {
            blocks["nin_shortcut"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, out_channels, {1, 1}));
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // x: [N, in_channels, h, w]
        // t_emb is always None
        auto norm1 = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm1"]);
        auto conv1 = std::dynamic_pointer_cast<Conv2d>(blocks["conv1"]);
        auto norm2 = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm2"]);
        auto conv2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv2"]);

        auto h = x;
        h      = norm1->forward(ctx, h);
        h      = ggml_silu_inplace(ctx->ggml_ctx, h);  // swish
        h      = conv1->forward(ctx, h);
        // return h;

        h = norm2->forward(ctx, h);
        h = ggml_silu_inplace(ctx->ggml_ctx, h);  // swish
        // dropout, skip for inference
        h = conv2->forward(ctx, h);

        // skip connection
        if (out_channels != in_channels) {
            auto nin_shortcut = std::dynamic_pointer_cast<Conv2d>(blocks["nin_shortcut"]);

            x = nin_shortcut->forward(ctx, x);  // [N, out_channels, h, w]
        }

        h = ggml_add(ctx->ggml_ctx, h, x);
        return h;  // [N, out_channels, h, w]
    }
};

class AttnBlock : public UnaryBlock {
protected:
    int64_t in_channels;
    bool use_linear;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") {
        auto iter = tensor_storage_map.find(prefix + "proj_out.weight");
        if (iter != tensor_storage_map.end()) {
            if (iter->second.n_dims == 4 && use_linear) {
                use_linear         = false;
                blocks["q"]        = std::make_shared<Conv2d>(in_channels, in_channels, std::pair{1, 1});
                blocks["k"]        = std::make_shared<Conv2d>(in_channels, in_channels, std::pair{1, 1});
                blocks["v"]        = std::make_shared<Conv2d>(in_channels, in_channels, std::pair{1, 1});
                blocks["proj_out"] = std::make_shared<Conv2d>(in_channels, in_channels, std::pair{1, 1});
            } else if (iter->second.n_dims == 2 && !use_linear) {
                use_linear         = true;
                blocks["q"]        = std::make_shared<Linear>(in_channels, in_channels);
                blocks["k"]        = std::make_shared<Linear>(in_channels, in_channels);
                blocks["v"]        = std::make_shared<Linear>(in_channels, in_channels);
                blocks["proj_out"] = std::make_shared<Linear>(in_channels, in_channels);
            }
        }
    }

public:
    AttnBlock(int64_t in_channels, bool use_linear)
        : in_channels(in_channels), use_linear(use_linear) {
        blocks["norm"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(in_channels));
        if (use_linear) {
            blocks["q"]        = std::shared_ptr<GGMLBlock>(new Linear(in_channels, in_channels));
            blocks["k"]        = std::shared_ptr<GGMLBlock>(new Linear(in_channels, in_channels));
            blocks["v"]        = std::shared_ptr<GGMLBlock>(new Linear(in_channels, in_channels));
            blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, in_channels));
        } else {
            blocks["q"]        = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, in_channels, {1, 1}));
            blocks["k"]        = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, in_channels, {1, 1}));
            blocks["v"]        = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, in_channels, {1, 1}));
            blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, in_channels, {1, 1}));
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // x: [N, in_channels, h, w]
        x = ggml_vae_require_f32_for_conv(ctx, x);
        auto norm     = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm"]);
        auto q_proj   = std::dynamic_pointer_cast<UnaryBlock>(blocks["q"]);
        auto k_proj   = std::dynamic_pointer_cast<UnaryBlock>(blocks["k"]);
        auto v_proj   = std::dynamic_pointer_cast<UnaryBlock>(blocks["v"]);
        auto proj_out = std::dynamic_pointer_cast<UnaryBlock>(blocks["proj_out"]);

        auto h_ = norm->forward(ctx, x);

        const int64_t n = h_->ne[3];
        const int64_t c = h_->ne[2];
        const int64_t h = h_->ne[1];
        const int64_t w = h_->ne[0];

        ggml_tensor* q;
        ggml_tensor* k;
        ggml_tensor* v;
        if (use_linear) {
            h_ = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h_, 1, 2, 0, 3));  // [N, h, w, in_channels]
            h_ = ggml_reshape_3d(ctx->ggml_ctx, h_, c, h * w, n);                        // [N, h * w, in_channels]

            q = q_proj->forward(ctx, h_);  // [N, h * w, in_channels]
            k = k_proj->forward(ctx, h_);  // [N, h * w, in_channels]
            v = v_proj->forward(ctx, h_);  // [N, h * w, in_channels]
        } else {
            q = q_proj->forward(ctx, h_);                                              // [N, in_channels, h, w]
            q = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, q, 1, 2, 0, 3));  // [N, h, w, in_channels]
            q = ggml_reshape_3d(ctx->ggml_ctx, q, c, h * w, n);                        // [N, h * w, in_channels]

            k = k_proj->forward(ctx, h_);                                              // [N, in_channels, h, w]
            k = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, k, 1, 2, 0, 3));  // [N, h, w, in_channels]
            k = ggml_reshape_3d(ctx->ggml_ctx, k, c, h * w, n);                        // [N, h * w, in_channels]

            v = v_proj->forward(ctx, h_);                                              // [N, in_channels, h, w]
            v = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, v, 1, 2, 0, 3));  // [N, h, w, in_channels]
            v = ggml_reshape_3d(ctx->ggml_ctx, v, c, h * w, n);                        // [N, h * w, in_channels]
        }

        h_ = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, 1, nullptr, false, ctx->flash_attn_enabled);

        if (use_linear) {
            h_ = proj_out->forward(ctx, h_);  // [N, h * w, in_channels]

            h_ = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h_, 1, 0, 2, 3));  // [N, in_channels, h * w]
            h_ = ggml_reshape_4d(ctx->ggml_ctx, h_, w, h, c, n);                         // [N, in_channels, h, w]
        } else {
            h_ = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h_, 1, 0, 2, 3));  // [N, in_channels, h * w]
            h_ = ggml_reshape_4d(ctx->ggml_ctx, h_, w, h, c, n);                         // [N, in_channels, h, w]

            h_ = proj_out->forward(ctx, h_);  // [N, in_channels, h, w]
        }

        h_ = ggml_add(ctx->ggml_ctx, h_, x);
        return ggml_vae_maybe_bf16_activation(ctx, h_);
    }
};

class AE3DConv : public Conv2d {
public:
    AE3DConv(int64_t in_channels,
             int64_t out_channels,
             std::pair<int, int> kernel_size,
             int video_kernel_size        = 3,
             std::pair<int, int> stride   = {1, 1},
             std::pair<int, int> padding  = {0, 0},
             std::pair<int, int> dilation = {1, 1},
             bool bias                    = true)
        : Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias) {
        int kernel_padding      = video_kernel_size / 2;
        blocks["time_mix_conv"] = std::shared_ptr<GGMLBlock>(new Conv3d(out_channels,
                                                                        out_channels,
                                                                        {video_kernel_size, 1, 1},
                                                                        {1, 1, 1},
                                                                        {kernel_padding, 0, 0}));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x) override {
        // timesteps always None
        // skip_video always False
        // x: [N, IC, IH, IW]
        // result: [N, OC, OH, OW]
        auto time_mix_conv = std::dynamic_pointer_cast<Conv3d>(blocks["time_mix_conv"]);

        x = Conv2d::forward(ctx, x);
        // timesteps = x.shape[0]
        // x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)
        // x = conv3d(x)
        // return rearrange(x, "b c t h w -> (b t) c h w")
        int64_t T = x->ne[3];
        int64_t B = x->ne[3] / T;
        int64_t C = x->ne[2];
        int64_t H = x->ne[1];
        int64_t W = x->ne[0];

        x = ggml_reshape_4d(ctx->ggml_ctx, x, W * H, C, T, B);                     // (b t) c h w -> b t c (h w)
        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // b t c (h w) -> b c t (h w)
        x = time_mix_conv->forward(ctx, x);                                        // [B, OC, T, OH * OW]
        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // b c t (h w) -> b t c (h w)
        x = ggml_reshape_4d(ctx->ggml_ctx, x, W, H, C, T * B);                     // b t c (h w) -> (b t) c h w
        return x;                                                                  // [B*T, OC, OH, OW]
    }
};

class VideoResnetBlock : public ResnetBlock {
protected:
    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        enum ggml_type wtype = get_type(prefix + "mix_factor", tensor_storage_map, GGML_TYPE_F32);
        params["mix_factor"] = ggml_new_tensor_1d(ctx, wtype, 1);
    }

    float get_alpha() {
        float alpha = ggml_ext_backend_tensor_get_f32(params["mix_factor"]);
        return sigmoid(alpha);
    }

public:
    VideoResnetBlock(int64_t in_channels,
                     int64_t out_channels,
                     int video_kernel_size = 3)
        : ResnetBlock(in_channels, out_channels) {
        // merge_strategy is always learned
        blocks["time_stack"] = std::shared_ptr<GGMLBlock>(new ResBlock(out_channels, 0, out_channels, {video_kernel_size, 1}, 3, false, true));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // x: [N, in_channels, h, w] aka [b*t, in_channels, h, w]
        // return: [N, out_channels, h, w] aka [b*t, out_channels, h, w]
        // t_emb is always None
        // skip_video is always False
        // timesteps is always None
        auto time_stack = std::dynamic_pointer_cast<ResBlock>(blocks["time_stack"]);

        x = ResnetBlock::forward(ctx, x);  // [N, out_channels, h, w]
        // return x;

        int64_t T = x->ne[3];
        int64_t B = x->ne[3] / T;
        int64_t C = x->ne[2];
        int64_t H = x->ne[1];
        int64_t W = x->ne[0];

        x          = ggml_reshape_4d(ctx->ggml_ctx, x, W * H, C, T, B);                     // (b t) c h w -> b t c (h w)
        x          = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // b t c (h w) -> b c t (h w)
        auto x_mix = x;

        x = time_stack->forward(ctx, x);  // b t c (h w)

        float alpha = get_alpha();
        x           = ggml_add(ctx->ggml_ctx,
                               ggml_ext_scale(ctx->ggml_ctx, x, alpha),
                               ggml_ext_scale(ctx->ggml_ctx, x_mix, 1.0f - alpha));

        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // b c t (h w) -> b t c (h w)
        x = ggml_reshape_4d(ctx->ggml_ctx, x, W, H, C, T * B);                     // b t c (h w) -> (b t) c h w

        return x;
    }
};

// ldm.modules.diffusionmodules.model.Encoder
class Encoder : public GGMLBlock {
protected:
    int ch                   = 128;
    std::vector<int> ch_mult = {1, 2, 4, 4};
    int num_res_blocks       = 2;
    int in_channels          = 3;
    int z_channels           = 4;
    bool double_z            = true;

public:
    Encoder(int ch,
            std::vector<int> ch_mult,
            int num_res_blocks,
            int in_channels,
            int z_channels,
            bool double_z              = true,
            bool use_linear_projection = false)
        : ch(ch),
          ch_mult(ch_mult),
          num_res_blocks(num_res_blocks),
          in_channels(in_channels),
          z_channels(z_channels),
          double_z(double_z) {
        blocks["conv_in"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, ch, {3, 3}, {1, 1}, {1, 1}));

        size_t num_resolutions = ch_mult.size();

        int block_in = 1;
        for (int i = 0; i < num_resolutions; i++) {
            if (i == 0) {
                block_in = ch;
            } else {
                block_in = ch * ch_mult[i - 1];
            }
            int block_out = ch * ch_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                std::string name = "down." + std::to_string(i) + ".block." + std::to_string(j);
                blocks[name]     = std::shared_ptr<GGMLBlock>(new ResnetBlock(block_in, block_out));
                block_in         = block_out;
            }
            if (i != num_resolutions - 1) {
                std::string name = "down." + std::to_string(i) + ".downsample";
                blocks[name]     = std::shared_ptr<GGMLBlock>(new DownSampleBlock(block_in, block_in, true));
            }
        }

        blocks["mid.block_1"] = std::shared_ptr<GGMLBlock>(new ResnetBlock(block_in, block_in));
        blocks["mid.attn_1"]  = std::shared_ptr<GGMLBlock>(new AttnBlock(block_in, use_linear_projection));
        blocks["mid.block_2"] = std::shared_ptr<GGMLBlock>(new ResnetBlock(block_in, block_in));

        blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(block_in));
        blocks["conv_out"] = std::shared_ptr<GGMLBlock>(new Conv2d(block_in, double_z ? z_channels * 2 : z_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    virtual ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [N, in_channels, h, w]

        auto conv_in     = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"]);
        auto mid_block_1 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_1"]);
        auto mid_attn_1  = std::dynamic_pointer_cast<AttnBlock>(blocks["mid.attn_1"]);
        auto mid_block_2 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_2"]);
        auto norm_out    = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm_out"]);
        auto conv_out    = std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"]);

        auto h = conv_in->forward(ctx, x);  // [N, ch, h, w]

        // downsampling
        size_t num_resolutions = ch_mult.size();
        for (int i = 0; i < num_resolutions; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                std::string name = "down." + std::to_string(i) + ".block." + std::to_string(j);
                auto down_block  = std::dynamic_pointer_cast<ResnetBlock>(blocks[name]);

                h = down_block->forward(ctx, h);
            }
            if (i != num_resolutions - 1) {
                std::string name = "down." + std::to_string(i) + ".downsample";
                auto down_sample = std::dynamic_pointer_cast<DownSampleBlock>(blocks[name]);

                h = down_sample->forward(ctx, h);
            }
        }

        // middle
        h = mid_block_1->forward(ctx, h);
        h = mid_attn_1->forward(ctx, h);
        h = mid_block_2->forward(ctx, h);  // [N, block_in, h, w]

        // end
        h = norm_out->forward(ctx, h);
        h = ggml_silu_inplace(ctx->ggml_ctx, h);  // nonlinearity/swish
        h = conv_out->forward(ctx, h);            // [N, z_channels*2, h, w]
        return h;
    }

    int stage_count() const {
        return static_cast<int>(ch_mult.size()) + 2;
    }

    ggml_tensor* forward_stage(GGMLRunnerContext* ctx, ggml_tensor* x, int stage) {
        auto conv_in     = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"]);
        auto mid_block_1 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_1"]);
        auto mid_attn_1  = std::dynamic_pointer_cast<AttnBlock>(blocks["mid.attn_1"]);
        auto mid_block_2 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_2"]);
        auto norm_out    = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm_out"]);
        auto conv_out    = std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"]);

        if (stage == 0) {
            return conv_in->forward(ctx, x);
        }

        size_t num_resolutions = ch_mult.size();
        if (stage >= 1 && stage <= static_cast<int>(num_resolutions)) {
            int i = stage - 1;
            auto h = x;
            for (int j = 0; j < num_res_blocks; j++) {
                std::string name = "down." + std::to_string(i) + ".block." + std::to_string(j);
                auto down_block  = std::dynamic_pointer_cast<ResnetBlock>(blocks[name]);
                h = down_block->forward(ctx, h);
            }
            if (i != static_cast<int>(num_resolutions) - 1) {
                std::string name = "down." + std::to_string(i) + ".downsample";
                auto down_sample = std::dynamic_pointer_cast<DownSampleBlock>(blocks[name]);
                h = down_sample->forward(ctx, h);
            }
            return h;
        }

        auto h = mid_block_1->forward(ctx, x);
        h = mid_attn_1->forward(ctx, h);
        h = mid_block_2->forward(ctx, h);
        h = norm_out->forward(ctx, h);
        h = ggml_silu_inplace(ctx->ggml_ctx, h);
        h = conv_out->forward(ctx, h);
        return h;
    }
};

// ldm.modules.diffusionmodules.model.Decoder
class Decoder : public GGMLBlock {
protected:
    int ch                   = 128;
    int out_ch               = 3;
    std::vector<int> ch_mult = {1, 2, 4, 4};
    int num_res_blocks       = 2;
    int z_channels           = 4;
    bool video_decoder       = false;
    int video_kernel_size    = 3;

    virtual std::shared_ptr<GGMLBlock> get_conv_out(int64_t in_channels,
                                                    int64_t out_channels,
                                                    std::pair<int, int> kernel_size,
                                                    std::pair<int, int> stride  = {1, 1},
                                                    std::pair<int, int> padding = {0, 0}) {
        if (video_decoder) {
            return std::shared_ptr<GGMLBlock>(new AE3DConv(in_channels, out_channels, kernel_size, video_kernel_size, stride, padding));
        } else {
            return std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, out_channels, kernel_size, stride, padding));
        }
    }

    virtual std::shared_ptr<GGMLBlock> get_resnet_block(int64_t in_channels,
                                                        int64_t out_channels) {
        if (video_decoder) {
            return std::shared_ptr<GGMLBlock>(new VideoResnetBlock(in_channels, out_channels, video_kernel_size));
        } else {
            return std::shared_ptr<GGMLBlock>(new ResnetBlock(in_channels, out_channels));
        }
    }

public:
    Decoder(int ch,
            int out_ch,
            std::vector<int> ch_mult,
            int num_res_blocks,
            int z_channels,
            bool use_linear_projection = false,
            bool video_decoder         = false,
            int video_kernel_size      = 3)
        : ch(ch),
          out_ch(out_ch),
          ch_mult(ch_mult),
          num_res_blocks(num_res_blocks),
          z_channels(z_channels),
          video_decoder(video_decoder),
          video_kernel_size(video_kernel_size) {
        int num_resolutions = static_cast<int>(ch_mult.size());
        int block_in        = ch * ch_mult[num_resolutions - 1];

        blocks["conv_in"] = std::shared_ptr<GGMLBlock>(new Conv2d(z_channels, block_in, {3, 3}, {1, 1}, {1, 1}));

        blocks["mid.block_1"] = get_resnet_block(block_in, block_in);
        blocks["mid.attn_1"]  = std::shared_ptr<GGMLBlock>(new AttnBlock(block_in, use_linear_projection));
        blocks["mid.block_2"] = get_resnet_block(block_in, block_in);

        for (int i = num_resolutions - 1; i >= 0; i--) {
            int mult      = ch_mult[i];
            int block_out = ch * mult;
            for (int j = 0; j < num_res_blocks + 1; j++) {
                std::string name = "up." + std::to_string(i) + ".block." + std::to_string(j);
                blocks[name]     = get_resnet_block(block_in, block_out);

                block_in = block_out;
            }
            if (i != 0) {
                std::string name = "up." + std::to_string(i) + ".upsample";
                blocks[name]     = std::shared_ptr<GGMLBlock>(new UpSampleBlock(block_in, block_in));
            }
        }

        blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(block_in));
        blocks["conv_out"] = get_conv_out(block_in, out_ch, {3, 3}, {1, 1}, {1, 1});
    }

    virtual ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* z) {
        // z: [N, z_channels, h, w]
        // alpha is always 0
        // merge_strategy is always learned
        // time_mode is always conv-only, so we need to replace conv_out_op/resnet_op to AE3DConv/VideoResBlock
        // AttnVideoBlock will not be used
        auto conv_in     = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"]);
        auto mid_block_1 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_1"]);
        auto mid_attn_1  = std::dynamic_pointer_cast<AttnBlock>(blocks["mid.attn_1"]);
        auto mid_block_2 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_2"]);
        auto norm_out    = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm_out"]);
        auto conv_out    = std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"]);

        // conv_in
        auto h = conv_in->forward(ctx, z);  // [N, block_in, h, w]

        // middle
        h = mid_block_1->forward(ctx, h);
        // return h;

        h = mid_attn_1->forward(ctx, h);
        h = mid_block_2->forward(ctx, h);  // [N, block_in, h, w]

        // upsampling
        int num_resolutions = static_cast<int>(ch_mult.size());
        for (int i = num_resolutions - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                std::string name = "up." + std::to_string(i) + ".block." + std::to_string(j);
                auto up_block    = std::dynamic_pointer_cast<ResnetBlock>(blocks[name]);

                h = up_block->forward(ctx, h);
            }
            if (i != 0) {
                std::string name = "up." + std::to_string(i) + ".upsample";
                auto up_sample   = std::dynamic_pointer_cast<UpSampleBlock>(blocks[name]);

                h = up_sample->forward(ctx, h);
            }
        }

        h = norm_out->forward(ctx, h);
        h = ggml_silu_inplace(ctx->ggml_ctx, h);  // nonlinearity/swish
        h = conv_out->forward(ctx, h);            // [N, out_ch, h*8, w*8]
        return h;
    }

    int stage_count() const {
        return static_cast<int>(ch_mult.size()) + 2;
    }

    ggml_tensor* forward_stage(GGMLRunnerContext* ctx, ggml_tensor* z, int stage) {
        auto conv_in     = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"]);
        auto mid_block_1 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_1"]);
        auto mid_attn_1  = std::dynamic_pointer_cast<AttnBlock>(blocks["mid.attn_1"]);
        auto mid_block_2 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_2"]);
        auto norm_out    = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm_out"]);
        auto conv_out    = std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"]);

        if (stage == 0) {
            auto h = conv_in->forward(ctx, z);
            h = mid_block_1->forward(ctx, h);
            h = mid_attn_1->forward(ctx, h);
            return mid_block_2->forward(ctx, h);
        }

        int num_resolutions = static_cast<int>(ch_mult.size());
        if (stage >= 1 && stage <= num_resolutions) {
            int i = num_resolutions - stage;
            auto h = z;
            for (int j = 0; j < num_res_blocks + 1; j++) {
                std::string name = "up." + std::to_string(i) + ".block." + std::to_string(j);
                auto up_block    = std::dynamic_pointer_cast<ResnetBlock>(blocks[name]);
                h = up_block->forward(ctx, h);
            }
            if (i != 0) {
                std::string name = "up." + std::to_string(i) + ".upsample";
                auto up_sample   = std::dynamic_pointer_cast<UpSampleBlock>(blocks[name]);
                h = up_sample->forward(ctx, h);
            }
            return h;
        }

        auto h = norm_out->forward(ctx, z);
        h = ggml_silu_inplace(ctx->ggml_ctx, h);
        h = conv_out->forward(ctx, h);
        return h;
    }
};

// ldm.models.autoencoder.AutoencoderKL
class AutoEncoderKLModel : public GGMLBlock {
protected:
    SDVersion version;
    bool decode_only       = true;
    bool use_video_decoder = false;
    bool use_quant         = true;
    int embed_dim          = 4;
    struct {
        int z_channels           = 4;
        int resolution           = 256;
        int in_channels          = 3;
        int out_ch               = 3;
        int ch                   = 128;
        std::vector<int> ch_mult = {1, 2, 4, 4};
        int num_res_blocks       = 2;
        bool double_z            = true;
    } dd_config;

    static std::string get_tensor_name(const std::string& prefix, const std::string& name) {
        return prefix.empty() ? name : prefix + "." + name;
    }

    void detect_decoder_ch(const String2TensorStorage& tensor_storage_map,
                           const std::string& prefix,
                           int& decoder_ch) {
        auto conv_in_iter = tensor_storage_map.find(get_tensor_name(prefix, "decoder.conv_in.weight"));
        if (conv_in_iter != tensor_storage_map.end() && conv_in_iter->second.n_dims >= 4 && conv_in_iter->second.ne[3] > 0) {
            int last_ch_mult             = dd_config.ch_mult.back();
            int64_t conv_in_out_channels = conv_in_iter->second.ne[3];
            if (last_ch_mult > 0 && conv_in_out_channels % last_ch_mult == 0) {
                decoder_ch = static_cast<int>(conv_in_out_channels / last_ch_mult);
                LOG_INFO("vae decoder: ch = %d", decoder_ch);
            } else {
                LOG_WARN("vae decoder: failed to infer ch from %s (%" PRId64 " / %d)",
                         get_tensor_name(prefix, "decoder.conv_in.weight").c_str(),
                         conv_in_out_channels,
                         last_ch_mult);
            }
        }
    }

public:
    AutoEncoderKLModel(SDVersion version                              = VERSION_SD1,
                       bool decode_only                               = true,
                       bool use_linear_projection                     = false,
                       bool use_video_decoder                         = false,
                       const String2TensorStorage& tensor_storage_map = {},
                       const std::string& prefix                      = "")
        : version(version), decode_only(decode_only), use_video_decoder(use_video_decoder) {
        if (sd_version_is_dit(version)) {
            if (sd_version_is_flux2(version)) {
                dd_config.z_channels = 32;
                embed_dim            = 32;
            } else {
                use_quant            = false;
                dd_config.z_channels = 16;
            }
        }
        if (use_video_decoder) {
            use_quant = false;
        }
        int decoder_ch = dd_config.ch;
        detect_decoder_ch(tensor_storage_map, prefix, decoder_ch);
        blocks["decoder"] = std::shared_ptr<GGMLBlock>(new Decoder(decoder_ch,
                                                                   dd_config.out_ch,
                                                                   dd_config.ch_mult,
                                                                   dd_config.num_res_blocks,
                                                                   dd_config.z_channels,
                                                                   use_linear_projection,
                                                                   use_video_decoder));
        if (use_quant) {
            blocks["post_quant_conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(dd_config.z_channels,
                                                                              embed_dim,
                                                                              {1, 1}));
        }
        if (!decode_only) {
            blocks["encoder"] = std::shared_ptr<GGMLBlock>(new Encoder(dd_config.ch,
                                                                       dd_config.ch_mult,
                                                                       dd_config.num_res_blocks,
                                                                       dd_config.in_channels,
                                                                       dd_config.z_channels,
                                                                       dd_config.double_z,
                                                                       use_linear_projection));
            if (use_quant) {
                int factor = dd_config.double_z ? 2 : 1;

                blocks["quant_conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(embed_dim * factor,
                                                                             dd_config.z_channels * factor,
                                                                             {1, 1}));
            }
        }
    }

    ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* z) {
        // z: [N, z_channels, h, w]
        if (sd_version_is_flux2(version)) {
            // [N, C*p*p, h, w] -> [N, C, h*p, w*p]
            int64_t p = 2;

            int64_t N = z->ne[3];
            int64_t C = z->ne[2] / p / p;
            int64_t h = z->ne[1];
            int64_t w = z->ne[0];
            int64_t H = h * p;
            int64_t W = w * p;

            z = ggml_reshape_4d(ctx->ggml_ctx, z, w * h, p * p, C, N);                           // [N, C, p*p, h*w]
            z = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, z, 1, 0, 2, 3));  // [N, C, h*w, p*p]
            z = ggml_reshape_4d(ctx->ggml_ctx, z, p, p, w, h * C * N);                           // [N*C*h, w, p, p]
            z = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, z, 0, 2, 1, 3));  // [N*C*h, p, w, p]
            z = ggml_reshape_4d(ctx->ggml_ctx, z, W, H, C, N);                                   // [N, C, h*p, w*p]
        }

        if (use_quant) {
            auto post_quant_conv = std::dynamic_pointer_cast<Conv2d>(blocks["post_quant_conv"]);
            z                    = post_quant_conv->forward(ctx, z);  // [N, z_channels, h, w]
        }
        auto decoder = std::dynamic_pointer_cast<Decoder>(blocks["decoder"]);

        ggml_set_name(z, "bench-start");
        auto h = decoder->forward(ctx, z);
        ggml_set_name(h, "bench-end");
        return h;
    }

    int decode_stage_count() {
        auto decoder = std::dynamic_pointer_cast<Decoder>(blocks["decoder"]);
        return decoder->stage_count();
    }

    ggml_tensor* decode_stage(GGMLRunnerContext* ctx, ggml_tensor* z, int stage) {
        if (stage == 0) {
            if (sd_version_is_flux2(version)) {
                int64_t p = 2;
                int64_t N = z->ne[3];
                int64_t C = z->ne[2] / p / p;
                int64_t h = z->ne[1];
                int64_t w = z->ne[0];
                int64_t H = h * p;
                int64_t W = w * p;

                z = ggml_reshape_4d(ctx->ggml_ctx, z, w * h, p * p, C, N);
                z = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, z, 1, 0, 2, 3));
                z = ggml_reshape_4d(ctx->ggml_ctx, z, p, p, w, h * C * N);
                z = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, z, 0, 2, 1, 3));
                z = ggml_reshape_4d(ctx->ggml_ctx, z, W, H, C, N);
            }
            if (use_quant) {
                auto post_quant_conv = std::dynamic_pointer_cast<Conv2d>(blocks["post_quant_conv"]);
                z = post_quant_conv->forward(ctx, z);
            }
        }
        auto decoder = std::dynamic_pointer_cast<Decoder>(blocks["decoder"]);
        return decoder->forward_stage(ctx, z, stage);
    }

    ggml_tensor* encode(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [N, in_channels, h, w]
        auto encoder = std::dynamic_pointer_cast<Encoder>(blocks["encoder"]);

        auto z = encoder->forward(ctx, x);  // [N, 2*z_channels, h/8, w/8]
        if (use_quant) {
            auto quant_conv = std::dynamic_pointer_cast<Conv2d>(blocks["quant_conv"]);
            z               = quant_conv->forward(ctx, z);  // [N, 2*embed_dim, h/8, w/8]
        }
        if (sd_version_is_flux2(version)) {
            z = ggml_ext_chunk(ctx->ggml_ctx, z, 2, 2)[0];

            // [N, C, H, W] -> [N, C*p*p, H/p, W/p]
            int64_t p = 2;
            int64_t N = z->ne[3];
            int64_t C = z->ne[2];
            int64_t H = z->ne[1];
            int64_t W = z->ne[0];
            int64_t h = H / p;
            int64_t w = W / p;

            z = ggml_reshape_4d(ctx->ggml_ctx, z, p, w, p, h * C * N);                 // [N*C*h, p, w, p]
            z = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, z, 0, 2, 1, 3));  // [N*C*h, w, p, p]
            z = ggml_reshape_4d(ctx->ggml_ctx, z, p * p, w * h, C, N);                 // [N, C, h*w, p*p]
            z = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, z, 1, 0, 2, 3));  // [N, C, p*p, h*w]
            z = ggml_reshape_4d(ctx->ggml_ctx, z, w, h, p * p * C, N);                 // [N, C*p*p, h*w]
        }
        return z;
    }

    int encode_stage_count() {
        auto encoder = std::dynamic_pointer_cast<Encoder>(blocks["encoder"]);
        return encoder->stage_count();
    }

    ggml_tensor* encode_stage(GGMLRunnerContext* ctx, ggml_tensor* x, int stage) {
        auto encoder = std::dynamic_pointer_cast<Encoder>(blocks["encoder"]);
        auto z = encoder->forward_stage(ctx, x, stage);
        if (stage == encode_stage_count() - 1) {
            if (use_quant) {
                auto quant_conv = std::dynamic_pointer_cast<Conv2d>(blocks["quant_conv"]);
                z = quant_conv->forward(ctx, z);
            }
            if (sd_version_is_flux2(version)) {
                z = ggml_ext_chunk(ctx->ggml_ctx, z, 2, 2)[0];

                int64_t p = 2;
                int64_t N = z->ne[3];
                int64_t C = z->ne[2];
                int64_t H = z->ne[1];
                int64_t W = z->ne[0];
                int64_t h = H / p;
                int64_t w = W / p;

                z = ggml_reshape_4d(ctx->ggml_ctx, z, p, w, p, h * C * N);
                z = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, z, 0, 2, 1, 3));
                z = ggml_reshape_4d(ctx->ggml_ctx, z, p * p, w * h, C, N);
                z = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, z, 1, 0, 2, 3));
                z = ggml_reshape_4d(ctx->ggml_ctx, z, w, h, p * p * C, N);
            }
        }
        return z;
    }

    int get_encoder_output_channels() {
        int factor = dd_config.double_z ? 2 : 1;
        if (sd_version_is_flux2(version)) {
            return dd_config.z_channels * 4;
        }
        return dd_config.z_channels * factor;
    }
};

struct AutoEncoderKL : public VAE {
    float scale_factor = 1.f;
    float shift_factor = 0.f;
    bool decode_only   = true;
    bool comfy_normal_enabled = false;
    float diffusion_to_vae_shift_input = 0.f;
    float vae_to_diffusion_shift_input = 0.f;
    std::vector<float> diffusion_to_vae_flux2_mean_input;
    std::vector<float> diffusion_to_vae_flux2_std_scaled_input;
    std::vector<float> vae_to_diffusion_flux2_mean_input;
    std::vector<float> vae_to_diffusion_flux2_inv_std_scaled_input;
    std::vector<float> vae_encode_noise_input;
    AutoEncoderKLModel ae;

    AutoEncoderKL(ggml_backend_t backend,
                  bool offload_params_to_cpu,
                  const String2TensorStorage& tensor_storage_map,
                  const std::string prefix,
                  bool decode_only       = false,
                  bool use_video_decoder = false,
                  SDVersion version      = VERSION_SD1)
        : decode_only(decode_only), VAE(version, backend, offload_params_to_cpu) {
        if (sd_version_is_sd1(version) || sd_version_is_sd2(version) || sd_version_is_marigold_iid(version)) {
            scale_factor = 0.18215f;
            shift_factor = 0.f;
        } else if (sd_version_is_sdxl(version)) {
            scale_factor = 0.13025f;
            shift_factor = 0.f;
        } else if (sd_version_is_sd3(version)) {
            scale_factor = 1.5305f;
            shift_factor = 0.0609f;
        } else if (sd_version_is_flux(version) || sd_version_is_z_image(version)) {
            scale_factor = 0.3611f;
            shift_factor = 0.1159f;
        } else if (sd_version_is_flux2(version)) {
            scale_factor = 1.0f;
            shift_factor = 0.f;
        }
        bool use_linear_projection = false;
        for (const auto& [name, tensor_storage] : tensor_storage_map) {
            if (!starts_with(name, prefix)) {
                continue;
            }
            if (ends_with(name, "attn_1.proj_out.weight")) {
                if (tensor_storage.n_dims == 2) {
                    use_linear_projection = true;
                }
                break;
            }
        }
        ae = AutoEncoderKLModel(version, decode_only, use_linear_projection, use_video_decoder, tensor_storage_map, prefix);
        ae.init(params_ctx, tensor_storage_map, prefix);
    }

    void set_conv2d_scale(float scale) override {
        std::vector<GGMLBlock*> blocks;
        ae.get_all_blocks(blocks);
        for (auto block : blocks) {
            if (block->get_desc() == "Conv2d") {
                auto conv_block = (Conv2d*)block;
                conv_block->set_scale(scale);
            }
        }
    }

    void set_comfy_normal_enabled(bool enabled) override {
        comfy_normal_enabled = enabled;
    }

    GGMLRunnerContext get_context() override {
        auto runner_ctx = VAE::get_context();
        runner_ctx.vae_bf16_activations_enabled =
            comfy_normal_enabled && env_flag_enabled("SDCPP_EXPERIMENTAL_VAE_BF16");
        return runner_ctx;
    }

    std::string get_desc() override {
        return "vae";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) override {
        ae.get_param_tensors(tensors, prefix);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& z_tensor, bool decode_graph) {
        ggml_cgraph* gf = ggml_new_graph(compute_ctx);
        ggml_tensor* z  = make_input(z_tensor);

        auto runner_ctx = get_context();

        ggml_tensor* out = decode_graph ? ae.decode(&runner_ctx, z) : ae.encode(&runner_ctx, z);
        if (runner_ctx.vae_bf16_activations_enabled && out != nullptr && out->type == GGML_TYPE_BF16) {
            out = ggml_cast(compute_ctx, out, GGML_TYPE_F32);
        }

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    ggml_cgraph* build_stage_graph(const sd::Tensor<float>& z_tensor, bool decode_graph, int stage) {
        ggml_cgraph* gf = ggml_new_graph(compute_ctx);
        ggml_tensor* z  = make_input(z_tensor);

        auto runner_ctx = get_context();
        ggml_tensor* out = decode_graph ? ae.decode_stage(&runner_ctx, z, stage)
                                        : ae.encode_stage(&runner_ctx, z, stage);
        const int final_stage = decode_graph ? ae.decode_stage_count() - 1 : ae.encode_stage_count() - 1;
        if (runner_ctx.vae_bf16_activations_enabled && stage == final_stage && out != nullptr && out->type == GGML_TYPE_BF16) {
            out = ggml_cast(compute_ctx, out, GGML_TYPE_F32);
        }

        ggml_build_forward_expand(gf, out);
        return gf;
    }

    ggml_cgraph* build_stage_graph(ggml_tensor* z, bool decode_graph, int stage) {
        ggml_cgraph* gf = ggml_new_graph(compute_ctx);
        auto runner_ctx = get_context();
        ggml_tensor* out = decode_graph ? ae.decode_stage(&runner_ctx, z, stage)
                                        : ae.encode_stage(&runner_ctx, z, stage);
        const int final_stage = decode_graph ? ae.decode_stage_count() - 1 : ae.encode_stage_count() - 1;
        if (runner_ctx.vae_bf16_activations_enabled && stage == final_stage && out != nullptr && out->type == GGML_TYPE_BF16) {
            out = ggml_cast(compute_ctx, out, GGML_TYPE_F32);
        }

        ggml_build_forward_expand(gf, out);
        return gf;
    }

    ggml_cgraph* build_stage_range_graph(const sd::Tensor<float>& z_tensor, bool decode_graph, int first_stage, int last_stage) {
        ggml_cgraph* gf = ggml_new_graph(compute_ctx);
        ggml_tensor* out = make_input(z_tensor);

        auto runner_ctx = get_context();
        for (int stage = first_stage; stage <= last_stage; ++stage) {
            out = decode_graph ? ae.decode_stage(&runner_ctx, out, stage)
                               : ae.encode_stage(&runner_ctx, out, stage);
        }
        const int final_stage = decode_graph ? ae.decode_stage_count() - 1 : ae.encode_stage_count() - 1;
        if (runner_ctx.vae_bf16_activations_enabled && last_stage == final_stage && out != nullptr && out->type == GGML_TYPE_BF16) {
            out = ggml_cast(compute_ctx, out, GGML_TYPE_F32);
        }

        ggml_build_forward_expand(gf, out);
        return gf;
    }

    ggml_cgraph* build_stage_range_graph(ggml_tensor* z, bool decode_graph, int first_stage, int last_stage) {
        ggml_cgraph* gf = ggml_new_graph(compute_ctx);
        auto runner_ctx = get_context();
        ggml_tensor* out = z;
        for (int stage = first_stage; stage <= last_stage; ++stage) {
            out = decode_graph ? ae.decode_stage(&runner_ctx, out, stage)
                               : ae.encode_stage(&runner_ctx, out, stage);
        }
        const int final_stage = decode_graph ? ae.decode_stage_count() - 1 : ae.encode_stage_count() - 1;
        if (runner_ctx.vae_bf16_activations_enabled && last_stage == final_stage && out != nullptr && out->type == GGML_TYPE_BF16) {
            out = ggml_cast(compute_ctx, out, GGML_TYPE_F32);
        }

        ggml_build_forward_expand(gf, out);
        return gf;
    }

    static int env_int_or_default(const char* name, int fallback) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return fallback;
        }
        char* end = nullptr;
        long parsed = std::strtol(value, &end, 10);
        if (end == value) {
            return fallback;
        }
        return static_cast<int>(parsed);
    }

    std::vector<std::pair<int, int>> stage_ranges(bool decode_graph) {
        const int stages = decode_graph ? ae.decode_stage_count() : ae.encode_stage_count();
        int merge_tail = 1;
        if (decode_graph && comfy_normal_enabled) {
            const char* disable_fuse_scale_env = std::getenv("SDCPP_DISABLE_VAE_FUSE_CONV_SCALE");
            const bool fuse_scale_disabled = disable_fuse_scale_env != nullptr &&
                                             disable_fuse_scale_env[0] != '\0' &&
                                             disable_fuse_scale_env[0] != '0';
            const int default_merge_tail = sd_version_is_sdxl(version) && !fuse_scale_disabled ? stages : 1;
            merge_tail = env_int_or_default("SDCPP_VAE_DECODE_TAIL_MERGE", default_merge_tail);
        }
        merge_tail = std::max(1, std::min(merge_tail, stages));

        std::vector<std::pair<int, int>> ranges;
        const int split = stages - merge_tail;
        for (int stage = 0; stage < split; ++stage) {
            ranges.push_back({stage, stage});
        }
        ranges.push_back({split, stages - 1});
        return ranges;
    }

    void log_stage_timing_if_enabled(bool decode_graph,
                                     size_t graph_index,
                                     const std::pair<int, int>& range,
                                     const sd_vae_memory_report_t& report,
                                     int64_t elapsed_ms,
                                     const char* path) {
        if (!env_flag_enabled("SDCPP_TRACE_VAE_TIMING")) {
            return;
        }
        LOG_INFO("[VAE] COMFY_NORMAL %s %s graph=%zu stage_range=%d-%d total_ms=%" PRId64 " workspace=%.2fMB largest=%.2fMB op=%s type=%s shape=%s",
                 decode_graph ? "decode" : "encode",
                 path == nullptr ? "stage" : path,
                 graph_index,
                 range.first,
                 range.second,
                 elapsed_ms,
                 report.planned_workspace_bytes / 1024.0 / 1024.0,
                 report.largest_tensor_bytes / 1024.0 / 1024.0,
                 report.largest_tensor_op,
                 report.largest_tensor_type,
                 report.largest_tensor_shape);
    }

    void prepare_flux2_diffusion_to_vae_inputs() {
        if (diffusion_to_vae_flux2_mean_input.size() == 128 &&
            diffusion_to_vae_flux2_std_scaled_input.size() == 128) {
            return;
        }
        sd::Tensor<float> dummy({1, 1, 128, 1});
        auto [mean_tensor, std_tensor] = get_latents_mean_std(dummy, 2);
        diffusion_to_vae_flux2_mean_input.assign(mean_tensor.data(), mean_tensor.data() + mean_tensor.numel());
        diffusion_to_vae_flux2_std_scaled_input.resize(static_cast<size_t>(std_tensor.numel()));
        for (int64_t i = 0; i < std_tensor.numel(); ++i) {
            diffusion_to_vae_flux2_std_scaled_input[static_cast<size_t>(i)] = std_tensor[i] / scale_factor;
        }
    }

    void prepare_flux2_vae_to_diffusion_inputs() {
        if (vae_to_diffusion_flux2_mean_input.size() == 128 &&
            vae_to_diffusion_flux2_inv_std_scaled_input.size() == 128) {
            return;
        }
        sd::Tensor<float> dummy({1, 1, 128, 1});
        auto [mean_tensor, std_tensor] = get_latents_mean_std(dummy, 2);
        vae_to_diffusion_flux2_mean_input.assign(mean_tensor.data(), mean_tensor.data() + mean_tensor.numel());
        vae_to_diffusion_flux2_inv_std_scaled_input.resize(static_cast<size_t>(std_tensor.numel()));
        for (int64_t i = 0; i < std_tensor.numel(); ++i) {
            vae_to_diffusion_flux2_inv_std_scaled_input[static_cast<size_t>(i)] = scale_factor / std_tensor[i];
        }
    }

    ggml_cgraph* build_diffusion_to_vae_latent_graph(ggml_tensor* z) {
        ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        ggml_tensor* out = nullptr;
        if (sd_version_is_flux2(version)) {
            prepare_flux2_diffusion_to_vae_inputs();

            ggml_tensor* std_scaled = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 1, 1, 128, 1);
            set_backend_tensor_data(std_scaled, diffusion_to_vae_flux2_std_scaled_input.data());
            out = ggml_mul(compute_ctx, z, std_scaled);

            ggml_tensor* mean_base = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 1, 1, 128, 1);
            set_backend_tensor_data(mean_base, diffusion_to_vae_flux2_mean_input.data());
            ggml_tensor* mean = ggml_repeat(compute_ctx, mean_base, out);
            out = ggml_add(compute_ctx, out, mean);
        } else {
            out = ggml_scale(compute_ctx, z, 1.0f / scale_factor);
            if (shift_factor != 0.0f) {
                diffusion_to_vae_shift_input = shift_factor;
                ggml_tensor* shift = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, 1);
                set_backend_tensor_data(shift, &diffusion_to_vae_shift_input);
                out = ggml_add1(compute_ctx, out, shift);
            }
        }
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    bool prepare_vae_encode_noise_input(ggml_tensor* vae_output, std::shared_ptr<RNG> rng) {
        vae_encode_noise_input.clear();
        if (vae_output == nullptr || rng == nullptr) {
            return false;
        }
        if (sd_version_is_flux2(version) || version == VERSION_SD1_PIX2PIX) {
            return true;
        }
        if (vae_output->ne[2] <= 0 || (vae_output->ne[2] % 2) != 0) {
            LOG_ERROR("VAE encode GPU latent transform expected even moments channels, got %" PRId64, vae_output->ne[2]);
            return false;
        }
        const int64_t numel = vae_output->ne[0] * vae_output->ne[1] * (vae_output->ne[2] / 2) * vae_output->ne[3];
        if (numel <= 0 || numel > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
            LOG_ERROR("VAE encode GPU latent transform invalid noise size: %" PRId64, numel);
            return false;
        }
        vae_encode_noise_input = rng->randn(static_cast<uint32_t>(numel));
        return static_cast<int64_t>(vae_encode_noise_input.size()) == numel;
    }

    ggml_cgraph* build_vae_output_to_diffusion_latent_graph(ggml_tensor* z) {
        ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        ggml_tensor* out = nullptr;
        if (sd_version_is_flux2(version)) {
            prepare_flux2_vae_to_diffusion_inputs();

            ggml_tensor* mean_base = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 1, 1, 128, 1);
            set_backend_tensor_data(mean_base, vae_to_diffusion_flux2_mean_input.data());
            ggml_tensor* mean = ggml_repeat(compute_ctx, mean_base, z);

            ggml_tensor* inv_std_scaled_base = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 1, 1, 128, 1);
            set_backend_tensor_data(inv_std_scaled_base, vae_to_diffusion_flux2_inv_std_scaled_input.data());
            ggml_tensor* inv_std_scaled = ggml_repeat(compute_ctx, inv_std_scaled_base, z);

            out = ggml_mul(compute_ctx, ggml_sub(compute_ctx, z, mean), inv_std_scaled);
        } else {
            ggml_tensor* latent = nullptr;
            if (version == VERSION_SD1_PIX2PIX) {
                latent = ggml_ext_chunk(compute_ctx, z, 2, 2)[0];
            } else {
                auto chunks = ggml_ext_chunk(compute_ctx, z, 2, 2);
                ggml_tensor* mean = chunks[0];
                ggml_tensor* logvar = ggml_clamp(compute_ctx, chunks[1], -30.0f, 20.0f);
                ggml_tensor* stddev = ggml_exp(compute_ctx, ggml_scale(compute_ctx, logvar, 0.5f));

                ggml_tensor* noise = ggml_new_tensor_4d(compute_ctx,
                                                        GGML_TYPE_F32,
                                                        mean->ne[0],
                                                        mean->ne[1],
                                                        mean->ne[2],
                                                        mean->ne[3]);
                set_backend_tensor_data(noise, vae_encode_noise_input.data());
                latent = ggml_add(compute_ctx, mean, ggml_mul(compute_ctx, stddev, noise));
            }

            out = latent;
            if (version != VERSION_SD1_PIX2PIX) {
                if (shift_factor != 0.0f) {
                    vae_to_diffusion_shift_input = -shift_factor;
                    ggml_tensor* shift = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, 1);
                    set_backend_tensor_data(shift, &vae_to_diffusion_shift_input);
                    out = ggml_add1(compute_ctx, out, shift);
                }
                out = ggml_scale(compute_ctx, out, scale_factor);
            }
        }

        ggml_build_forward_expand(gf, out);
        return gf;
    }

    static void merge_stage_report(sd_vae_memory_report_t* aggregate, const sd_vae_memory_report_t& stage_report) {
        aggregate->graph_count += std::max<uint32_t>(stage_report.graph_count, 1);
        aggregate->stage_count += 1;
        aggregate->used_im2col = aggregate->used_im2col || stage_report.used_im2col;
        aggregate->used_direct_conv = aggregate->used_direct_conv || stage_report.used_direct_conv;
        aggregate->compact_activation_storage = aggregate->compact_activation_storage || stage_report.compact_activation_storage;
        aggregate->estimated_peak_bytes = std::max(aggregate->estimated_peak_bytes, stage_report.estimated_peak_bytes);
        aggregate->planned_workspace_bytes = std::max(aggregate->planned_workspace_bytes, stage_report.planned_workspace_bytes);
        if (stage_report.largest_tensor_bytes > aggregate->largest_tensor_bytes) {
            aggregate->largest_tensor_bytes = stage_report.largest_tensor_bytes;
            std::snprintf(aggregate->largest_tensor_op, sizeof(aggregate->largest_tensor_op), "%s", stage_report.largest_tensor_op);
            std::snprintf(aggregate->largest_tensor_type, sizeof(aggregate->largest_tensor_type), "%s", stage_report.largest_tensor_type);
            std::snprintf(aggregate->largest_tensor_shape, sizeof(aggregate->largest_tensor_shape), "%s", stage_report.largest_tensor_shape);
        }
    }

    static void record_stage_output(sd_vae_memory_report_t* aggregate,
                                    int stage,
                                    const BackendTensorHandle* handle,
                                    bool boundary_input_was_device) {
        if (handle == nullptr || handle->empty()) {
            return;
        }
        aggregate->device_resident_stages = aggregate->device_resident_stages || !ggml_backend_buffer_is_host(handle->buffer);
        if (stage >= 0 && stage < 16) {
            std::snprintf(aggregate->stage_output_dtype[stage],
                          sizeof(aggregate->stage_output_dtype[stage]),
                          "%s",
                          ggml_type_name(handle->tensor->type));
            std::snprintf(aggregate->stage_output_backend[stage],
                          sizeof(aggregate->stage_output_backend[stage]),
                          "%s",
                          ggml_backend_buffer_name(handle->buffer));
        }
        if (boundary_input_was_device) {
            if (ggml_backend_buffer_is_host(handle->buffer)) {
                aggregate->stage_boundary_host_copies += 1;
            } else {
                aggregate->stage_boundary_device_copies += 1;
            }
        }
    }

    sd::Tensor<float> _compute_staged(const int n_threads,
                                      const sd::Tensor<float>& z,
                                      bool decode_graph) {
        GGML_ASSERT(!decode_only || decode_graph);
        const auto ranges = stage_ranges(decode_graph);
        std::unique_ptr<BackendTensorHandle> current;
        sd_vae_memory_report_t aggregate = {};
        aggregate.used_direct_conv = conv2d_direct_enabled;
        for (size_t graph_index = 0; graph_index < ranges.size(); ++graph_index) {
            const auto range = ranges[graph_index];
            auto get_graph = [&]() -> ggml_cgraph* {
                if (graph_index == 0) {
                    return build_stage_range_graph(z, decode_graph, range.first, range.second);
                }
                return build_stage_range_graph(current->tensor, decode_graph, range.first, range.second);
            };
            int64_t stage_t0 = ggml_time_ms();
            auto stage_output = compute_to_backend_handle(get_graph, n_threads, "vae_comfy_normal_stage");
            sd_vae_memory_report_t stage_report = get_last_graph_report();
            int64_t stage_t1 = ggml_time_ms();
            log_stage_timing_if_enabled(decode_graph, graph_index, range, stage_report, stage_t1 - stage_t0, "cpu-output");
            merge_stage_report(&aggregate, stage_report);
            if (stage_output == nullptr || stage_output->empty()) {
                return {};
            }
            record_stage_output(&aggregate, static_cast<int>(graph_index), stage_output.get(), graph_index > 0);
            current = std::move(stage_output);
        }
        aggregate.device_resident_stages = aggregate.stage_boundary_host_copies == 0 && !ggml_backend_buffer_is_host(current->buffer);
        LOG_INFO("[VAE] COMFY_NORMAL stage boundaries: device_resident=%s host_copies=%u device_copies=%u dtype_promotions=%u",
                 aggregate.device_resident_stages ? "true" : "false",
                 aggregate.stage_boundary_host_copies,
                 aggregate.stage_boundary_device_copies,
                 aggregate.stage_boundary_dtype_promotions);
        last_graph_report = aggregate;
        return take_or_empty(materialize_backend_tensor<float>(current.get(), z.dim()));
    }

    std::unique_ptr<GgmlBackendTensorResource> _compute_staged_resource(const int n_threads,
                                                                        const sd::Tensor<float>& z,
                                                                        bool decode_graph) {
        GGML_ASSERT(!decode_only || decode_graph);
        const auto ranges = stage_ranges(decode_graph);
        std::unique_ptr<BackendTensorHandle> current;
        std::unique_ptr<GgmlBackendTensorResource> final_resource;
        sd_vae_memory_report_t aggregate = {};
        aggregate.used_direct_conv = conv2d_direct_enabled;
        for (size_t graph_index = 0; graph_index < ranges.size(); ++graph_index) {
            const auto range = ranges[graph_index];
            auto get_graph = [&]() -> ggml_cgraph* {
                if (graph_index == 0) {
                    return build_stage_range_graph(z, decode_graph, range.first, range.second);
                }
                return build_stage_range_graph(current->tensor, decode_graph, range.first, range.second);
            };
            int64_t stage_t0 = ggml_time_ms();
            if (graph_index + 1 == ranges.size()) {
                final_resource = compute_to_backend_resource_handle(get_graph, n_threads, "vae_comfy_normal_gpu_output");
                sd_vae_memory_report_t stage_report = get_last_graph_report();
                int64_t stage_t1 = ggml_time_ms();
                log_stage_timing_if_enabled(decode_graph, graph_index, range, stage_report, stage_t1 - stage_t0, "gpu-output");
                merge_stage_report(&aggregate, stage_report);
                if (final_resource == nullptr || final_resource->empty()) {
                    return nullptr;
                }
                BackendTensorHandle final_view;
                final_view.tensor = final_resource->tensor;
                final_view.buffer = final_resource->buffer;
                record_stage_output(&aggregate, static_cast<int>(graph_index), &final_view, graph_index > 0);
                final_view.tensor = nullptr;
                final_view.buffer = nullptr;
            } else {
                auto stage_output = compute_to_backend_handle(get_graph, n_threads, "vae_comfy_normal_stage");
                sd_vae_memory_report_t stage_report = get_last_graph_report();
                int64_t stage_t1 = ggml_time_ms();
                log_stage_timing_if_enabled(decode_graph, graph_index, range, stage_report, stage_t1 - stage_t0, "stage");
                merge_stage_report(&aggregate, stage_report);
                if (stage_output == nullptr || stage_output->empty()) {
                    return nullptr;
                }
                record_stage_output(&aggregate, static_cast<int>(graph_index), stage_output.get(), graph_index > 0);
                current = std::move(stage_output);
            }
        }
        aggregate.device_resident_stages = aggregate.stage_boundary_host_copies == 0 &&
                                           final_resource != nullptr &&
                                           final_resource->buffer != nullptr &&
                                           !ggml_backend_buffer_is_host(final_resource->buffer);
        LOG_INFO("[VAE] COMFY_NORMAL stage boundaries: device_resident=%s host_copies=%u device_copies=%u dtype_promotions=%u",
                 aggregate.device_resident_stages ? "true" : "false",
                 aggregate.stage_boundary_host_copies,
                 aggregate.stage_boundary_device_copies,
                 aggregate.stage_boundary_dtype_promotions);
        last_graph_report = aggregate;
        return final_resource;
    }

    std::unique_ptr<GgmlBackendTensorResource> _compute_staged_resource_from_backend_tensor(const int n_threads,
                                                                                           ggml_tensor* z,
                                                                                           bool decode_graph) {
        GGML_ASSERT(!decode_only || decode_graph);
        const auto ranges = stage_ranges(decode_graph);
        std::unique_ptr<BackendTensorHandle> current;
        std::unique_ptr<GgmlBackendTensorResource> final_resource;
        sd_vae_memory_report_t aggregate = {};
        aggregate.used_direct_conv = conv2d_direct_enabled;
        for (size_t graph_index = 0; graph_index < ranges.size(); ++graph_index) {
            const auto range = ranges[graph_index];
            auto get_graph = [&]() -> ggml_cgraph* {
                if (graph_index == 0) {
                    return build_stage_range_graph(z, decode_graph, range.first, range.second);
                }
                return build_stage_range_graph(current->tensor, decode_graph, range.first, range.second);
            };
            int64_t stage_t0 = ggml_time_ms();
            if (graph_index + 1 == ranges.size()) {
                final_resource = compute_to_backend_resource_handle(get_graph, n_threads, "vae_comfy_normal_gpu_output");
                sd_vae_memory_report_t stage_report = get_last_graph_report();
                int64_t stage_t1 = ggml_time_ms();
                log_stage_timing_if_enabled(decode_graph, graph_index, range, stage_report, stage_t1 - stage_t0, "gpu-output");
                merge_stage_report(&aggregate, stage_report);
                if (final_resource == nullptr || final_resource->empty()) {
                    return nullptr;
                }
                BackendTensorHandle final_view;
                final_view.tensor = final_resource->tensor;
                final_view.buffer = final_resource->buffer;
                record_stage_output(&aggregate, static_cast<int>(graph_index), &final_view, graph_index > 0);
                final_view.tensor = nullptr;
                final_view.buffer = nullptr;
            } else {
                auto stage_output = compute_to_backend_handle(get_graph, n_threads, "vae_comfy_normal_stage");
                sd_vae_memory_report_t stage_report = get_last_graph_report();
                int64_t stage_t1 = ggml_time_ms();
                log_stage_timing_if_enabled(decode_graph, graph_index, range, stage_report, stage_t1 - stage_t0, "stage");
                merge_stage_report(&aggregate, stage_report);
                if (stage_output == nullptr || stage_output->empty()) {
                    return nullptr;
                }
                record_stage_output(&aggregate, static_cast<int>(graph_index), stage_output.get(), graph_index > 0);
                current = std::move(stage_output);
            }
        }
        aggregate.device_resident_stages = aggregate.stage_boundary_host_copies == 0 &&
                                           final_resource != nullptr &&
                                           final_resource->buffer != nullptr &&
                                           !ggml_backend_buffer_is_host(final_resource->buffer);
        LOG_INFO("[VAE] COMFY_NORMAL stage boundaries: device_resident=%s host_copies=%u device_copies=%u dtype_promotions=%u",
                 aggregate.device_resident_stages ? "true" : "false",
                 aggregate.stage_boundary_host_copies,
                 aggregate.stage_boundary_device_copies,
                 aggregate.stage_boundary_dtype_promotions);
        last_graph_report = aggregate;
        return final_resource;
    }

    sd::Tensor<float> _compute(const int n_threads,
                               const sd::Tensor<float>& z,
                               bool decode_graph) override {
        if (comfy_normal_enabled) {
            return _compute_staged(n_threads, z, decode_graph);
        }
        GGML_ASSERT(!decode_only || decode_graph);
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(z, decode_graph);
        };
        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), z.dim());
    }

    std::unique_ptr<GgmlBackendTensorResource> decode_to_backend_resource(int n_threads,
                                                                          const sd::Tensor<float>& x,
                                                                          sd_tiling_params_t tiling_params,
                                                                          bool circular_x = false,
                                                                          bool circular_y = false) override {
        SD_UNUSED(circular_x);
        SD_UNUSED(circular_y);
        if (!comfy_normal_enabled || tiling_params.enabled) {
            return nullptr;
        }
        int64_t t0 = ggml_time_ms();
        auto output = _compute_staged_resource(n_threads, x, true);
        if (output == nullptr || output->empty()) {
            LOG_ERROR("vae decode GPU output compute failed");
            return nullptr;
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing vae decode GPU output graph completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return output;
    }

    std::unique_ptr<GgmlBackendTensorResource> encode_to_backend_resource(int n_threads,
                                                                          const sd::Tensor<float>& x,
                                                                          sd_tiling_params_t tiling_params,
                                                                          std::shared_ptr<RNG> rng,
                                                                          bool circular_x = false,
                                                                          bool circular_y = false) override {
        SD_UNUSED(circular_x);
        SD_UNUSED(circular_y);
        if (!comfy_normal_enabled || tiling_params.enabled || decode_only) {
            return nullptr;
        }
        int64_t t0 = ggml_time_ms();
        sd::Tensor<float> input = x;
        if (scale_input) {
            scale_tensor_to_minus1_1(&input);
        }
        auto vae_output = _compute_staged_resource(n_threads, input, false);
        sd_vae_memory_report_t aggregate = get_last_graph_report();
        if (vae_output == nullptr || vae_output->empty()) {
            LOG_ERROR("vae encode GPU output compute failed");
            return nullptr;
        }
        if (!prepare_vae_encode_noise_input(vae_output->tensor, rng)) {
            return nullptr;
        }

        auto diffusion_latent = compute_to_backend_resource_handle(
            [&]() -> ggml_cgraph* {
                return build_vae_output_to_diffusion_latent_graph(vae_output->tensor);
            },
            n_threads,
            "vae_encode_diffusion_latent_gpu");
        sd_vae_memory_report_t transform_report = get_last_graph_report();
        const int transform_stage = static_cast<int>(aggregate.stage_count);
        merge_stage_report(&aggregate, transform_report);
        if (diffusion_latent == nullptr || diffusion_latent->empty()) {
            LOG_ERROR("VAE encode GPU latent transform failed");
            return nullptr;
        }
        BackendTensorHandle final_view;
        final_view.tensor = diffusion_latent->tensor;
        final_view.buffer = diffusion_latent->buffer;
        record_stage_output(&aggregate, transform_stage, &final_view, true);
        final_view.tensor = nullptr;
        final_view.buffer = nullptr;
        aggregate.device_resident_stages = aggregate.stage_boundary_host_copies == 0 &&
                                           diffusion_latent->buffer != nullptr &&
                                           !ggml_backend_buffer_is_host(diffusion_latent->buffer);
        LOG_INFO("[VAE] COMFY_NORMAL encode latent handoff: device_resident=%s host_copies=%u device_copies=%u dtype_promotions=%u",
                 aggregate.device_resident_stages ? "true" : "false",
                 aggregate.stage_boundary_host_copies,
                 aggregate.stage_boundary_device_copies,
                 aggregate.stage_boundary_dtype_promotions);
        last_graph_report = aggregate;
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing VAE encode GPU latent completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return diffusion_latent;
    }

    std::unique_ptr<GgmlBackendTensorResource> decode_latent_resource_to_backend_resource(int n_threads,
                                                                                         const GgmlBackendTensorResource* x,
                                                                                         sd_tiling_params_t tiling_params,
                                                                                         bool circular_x = false,
                                                                                         bool circular_y = false) override {
        SD_UNUSED(circular_x);
        SD_UNUSED(circular_y);
        if (!comfy_normal_enabled || tiling_params.enabled || x == nullptr || x->empty()) {
            return nullptr;
        }
        int64_t t0 = ggml_time_ms();
        auto vae_latent = compute_to_backend_resource_handle(
            [&]() -> ggml_cgraph* {
                return build_diffusion_to_vae_latent_graph(x->tensor);
            },
            n_threads,
            "vae_latent_from_gpu_diffusion_latent");
        if (vae_latent == nullptr || vae_latent->empty()) {
            LOG_ERROR("GPU latent diffusion-to-VAE transform failed");
            return nullptr;
        }
        auto output = _compute_staged_resource_from_backend_tensor(n_threads, vae_latent->tensor, true);
        if (output == nullptr || output->empty()) {
            LOG_ERROR("GPU latent VAE decode compute failed");
            return nullptr;
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing VAE decode from GPU latent completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return output;
    }

    bool estimate_memory_report(const sd::Tensor<float>& z,
                                bool decode_graph,
                                sd_vae_memory_report_t* report) override {
        if (comfy_normal_enabled) {
            const int stages = decode_graph ? ae.decode_stage_count() : ae.encode_stage_count();
            std::unique_ptr<BackendTensorHandle> current;
            sd_vae_memory_report_t aggregate = {};
            aggregate.used_direct_conv = conv2d_direct_enabled;
            for (int stage = 0; stage < stages; ++stage) {
                auto get_graph = [&]() -> ggml_cgraph* {
                    if (stage == 0) {
                        return build_stage_graph(z, decode_graph, stage);
                    }
                    return build_stage_graph(current->tensor, decode_graph, stage);
                };
                auto stage_output = compute_to_backend_handle(get_graph, 1, "vae_comfy_normal_estimate_stage");
                merge_stage_report(&aggregate, get_last_graph_report());
                if (stage_output == nullptr || stage_output->empty()) {
                    return false;
                }
                record_stage_output(&aggregate, stage, stage_output.get(), stage > 0);
                current = std::move(stage_output);
            }
            aggregate.device_resident_stages = aggregate.stage_boundary_host_copies == 0 &&
                                               current != nullptr &&
                                               !ggml_backend_buffer_is_host(current->buffer);
            last_graph_report = aggregate;
            if (report != nullptr) {
                *report = aggregate;
            }
            return true;
        }
        GGML_ASSERT(!decode_only || decode_graph);
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(z, decode_graph);
        };
        if (!offload_params_to_runtime_backend()) {
            LOG_ERROR("%s offload params to runtime backend failed", get_desc().c_str());
            return false;
        }
        if (!alloc_compute_buffer(get_graph)) {
            LOG_ERROR("%s alloc compute buffer failed while estimating VAE memory", get_desc().c_str());
            return false;
        }
        if (report != nullptr) {
            *report = get_last_graph_report();
        }
        free_compute_buffer();
        return true;
    }

    sd::Tensor<float> gaussian_latent_sample(const sd::Tensor<float>& moments, std::shared_ptr<RNG> rng) {
        // ldm.modules.distributions.distributions.DiagonalGaussianDistribution.sample
        auto chunks               = sd::ops::chunk(moments, 2, 2);
        const auto& mean          = chunks[0];
        const auto& logvar        = chunks[1];
        sd::Tensor<float> stddev  = sd::ops::exp(0.5f * sd::ops::clamp(logvar, -30.0f, 20.0f));
        sd::Tensor<float> noise   = sd::Tensor<float>::randn_like(mean, rng);
        sd::Tensor<float> latents = mean + stddev * noise;
        return latents;
    }

    sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) override {
        if (sd_version_is_flux2(version)) {
            return vae_output;
        } else if (version == VERSION_SD1_PIX2PIX || sd_version_is_marigold_iid(version)) {
            return sd::ops::chunk(vae_output, 2, 2)[0];
        } else {
            return gaussian_latent_sample(vae_output, rng);
        }
    }

    std::pair<sd::Tensor<float>, sd::Tensor<float>> get_latents_mean_std(const sd::Tensor<float>& latents, int channel_dim) {
        GGML_ASSERT(channel_dim >= 0 && static_cast<size_t>(channel_dim) < static_cast<size_t>(latents.dim()));
        if (sd_version_is_flux2(version)) {
            GGML_ASSERT(latents.shape()[channel_dim] == 128);
            std::vector<int64_t> stats_shape(static_cast<size_t>(latents.dim()), 1);
            stats_shape[static_cast<size_t>(channel_dim)] = latents.shape()[channel_dim];

            auto mean_tensor = sd::Tensor<float>::from_vector({-0.0676f, -0.0715f, -0.0753f, -0.0745f, 0.0223f, 0.0180f, 0.0142f, 0.0184f,
                                                               -0.0001f, -0.0063f, -0.0002f, -0.0031f, -0.0272f, -0.0281f, -0.0276f, -0.0290f,
                                                               -0.0769f, -0.0672f, -0.0902f, -0.0892f, 0.0168f, 0.0152f, 0.0079f, 0.0086f,
                                                               0.0083f, 0.0015f, 0.0003f, -0.0043f, -0.0439f, -0.0419f, -0.0438f, -0.0431f,
                                                               -0.0102f, -0.0132f, -0.0066f, -0.0048f, -0.0311f, -0.0306f, -0.0279f, -0.0180f,
                                                               0.0030f, 0.0015f, 0.0126f, 0.0145f, 0.0347f, 0.0338f, 0.0337f, 0.0283f,
                                                               0.0020f, 0.0047f, 0.0047f, 0.0050f, 0.0123f, 0.0081f, 0.0081f, 0.0146f,
                                                               0.0681f, 0.0679f, 0.0767f, 0.0732f, -0.0462f, -0.0474f, -0.0392f, -0.0511f,
                                                               -0.0528f, -0.0477f, -0.0470f, -0.0517f, -0.0317f, -0.0316f, -0.0345f, -0.0283f,
                                                               0.0510f, 0.0445f, 0.0578f, 0.0458f, -0.0412f, -0.0458f, -0.0487f, -0.0467f,
                                                               -0.0088f, -0.0106f, -0.0088f, -0.0046f, -0.0376f, -0.0432f, -0.0436f, -0.0499f,
                                                               0.0118f, 0.0166f, 0.0203f, 0.0279f, 0.0113f, 0.0129f, 0.0016f, 0.0072f,
                                                               -0.0118f, -0.0018f, -0.0141f, -0.0054f, -0.0091f, -0.0138f, -0.0145f, -0.0187f,
                                                               0.0323f, 0.0305f, 0.0259f, 0.0300f, 0.0540f, 0.0614f, 0.0495f, 0.0590f,
                                                               -0.0511f, -0.0603f, -0.0478f, -0.0524f, -0.0227f, -0.0274f, -0.0154f, -0.0255f,
                                                               -0.0572f, -0.0565f, -0.0518f, -0.0496f, 0.0116f, 0.0054f, 0.0163f, 0.0104f});
            mean_tensor.reshape_(stats_shape);
            auto std_tensor = sd::Tensor<float>::from_vector({1.8029f, 1.7786f, 1.7868f, 1.7837f, 1.7717f, 1.7590f, 1.7610f, 1.7479f,
                                                              1.7336f, 1.7373f, 1.7340f, 1.7343f, 1.8626f, 1.8527f, 1.8629f, 1.8589f,
                                                              1.7593f, 1.7526f, 1.7556f, 1.7583f, 1.7363f, 1.7400f, 1.7355f, 1.7394f,
                                                              1.7342f, 1.7246f, 1.7392f, 1.7304f, 1.7551f, 1.7513f, 1.7559f, 1.7488f,
                                                              1.8449f, 1.8454f, 1.8550f, 1.8535f, 1.8240f, 1.7813f, 1.7854f, 1.7945f,
                                                              1.8047f, 1.7876f, 1.7695f, 1.7676f, 1.7782f, 1.7667f, 1.7925f, 1.7848f,
                                                              1.7579f, 1.7407f, 1.7483f, 1.7368f, 1.7961f, 1.7998f, 1.7920f, 1.7925f,
                                                              1.7780f, 1.7747f, 1.7727f, 1.7749f, 1.7526f, 1.7447f, 1.7657f, 1.7495f,
                                                              1.7775f, 1.7720f, 1.7813f, 1.7813f, 1.8162f, 1.8013f, 1.8023f, 1.8033f,
                                                              1.7527f, 1.7331f, 1.7563f, 1.7482f, 1.7610f, 1.7507f, 1.7681f, 1.7613f,
                                                              1.7665f, 1.7545f, 1.7828f, 1.7726f, 1.7896f, 1.7999f, 1.7864f, 1.7760f,
                                                              1.7613f, 1.7625f, 1.7560f, 1.7577f, 1.7783f, 1.7671f, 1.7810f, 1.7799f,
                                                              1.7201f, 1.7068f, 1.7265f, 1.7091f, 1.7793f, 1.7578f, 1.7502f, 1.7455f,
                                                              1.7587f, 1.7500f, 1.7525f, 1.7362f, 1.7616f, 1.7572f, 1.7444f, 1.7430f,
                                                              1.7509f, 1.7610f, 1.7634f, 1.7612f, 1.7254f, 1.7135f, 1.7321f, 1.7226f,
                                                              1.7664f, 1.7624f, 1.7718f, 1.7664f, 1.7457f, 1.7441f, 1.7569f, 1.7530f});
            std_tensor.reshape_(stats_shape);
            return {std::move(mean_tensor), std::move(std_tensor)};
        } else {
            GGML_ABORT("unknown version %d", version);
        }
    }

    sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
        if (sd_version_is_flux2(version)) {
            int channel_dim                = 2;
            auto [mean_tensor, std_tensor] = get_latents_mean_std(latents, channel_dim);
            return (latents * std_tensor) / scale_factor + mean_tensor;
        }
        return (latents / scale_factor) + shift_factor;
    }

    sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
        if (sd_version_is_flux2(version)) {
            int channel_dim                = 2;
            auto [mean_tensor, std_tensor] = get_latents_mean_std(latents, channel_dim);
            return ((latents - mean_tensor) * scale_factor) / std_tensor;
        }
        return (latents - shift_factor) * scale_factor;
    }

    int get_encoder_output_channels(int input_channels) {
        return ae.get_encoder_output_channels();
    }

    void test() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
        params.mem_buffer = nullptr;
        params.no_alloc   = false;

        ggml_context* ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);

        {
            // CPU, x{1, 3, 64, 64}: Pass
            // CUDA, x{1, 3, 64, 64}: Pass, but sill get wrong result for some image, may be due to interlnal nan
            // CPU, x{2, 3, 64, 64}: Wrong result
            // CUDA, x{2, 3, 64, 64}: Wrong result, and different from CPU result
            sd::Tensor<float> x({64, 64, 3, 2});
            x.fill_(0.5f);
            print_sd_tensor(x);
            sd::Tensor<float> out;

            int64_t t0   = ggml_time_ms();
            auto out_opt = _compute(8, x, false);
            int64_t t1   = ggml_time_ms();

            GGML_ASSERT(!out_opt.empty());
            out = std::move(out_opt);
            print_sd_tensor(out);
            LOG_DEBUG("encode test done in %lldms", t1 - t0);
        }

        if (false) {
            // CPU, z{1, 4, 8, 8}: Pass
            // CUDA, z{1, 4, 8, 8}: Pass
            // CPU, z{3, 4, 8, 8}: Wrong result
            // CUDA, z{3, 4, 8, 8}: Wrong result, and different from CPU result
            sd::Tensor<float> z({8, 8, 4, 1});
            z.fill_(0.5f);
            print_sd_tensor(z);
            sd::Tensor<float> out;

            int64_t t0   = ggml_time_ms();
            auto out_opt = _compute(8, z, true);
            int64_t t1   = ggml_time_ms();

            GGML_ASSERT(!out_opt.empty());
            out = std::move(out_opt);
            print_sd_tensor(out);
            LOG_DEBUG("decode test done in %lldms", t1 - t0);
        }
    };
};

#endif  // __AUTO_ENCODER_KL_HPP__
