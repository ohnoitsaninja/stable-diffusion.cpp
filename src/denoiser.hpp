#ifndef __DENOISER_HPP__
#define __DENOISER_HPP__

#include <cmath>
#include <utility>

#include "ggml_extend.hpp"
#include "gits_noise.inl"
#include "tensor.hpp"

/*================================================= CompVisDenoiser ==================================================*/

// Ref: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py

#define TIMESTEPS 1000
#define FLUX_TIMESTEPS 1000

struct SigmaScheduler {
    typedef std::function<float(float)> t_to_sigma_t;

    virtual std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) = 0;
};

struct DiscreteScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        std::vector<float> result;

        int t_max = TIMESTEPS - 1;

        if (n == 0) {
            return result;
        } else if (n == 1) {
            result.push_back(t_to_sigma((float)t_max));
            result.push_back(0);
            return result;
        }

        float step = static_cast<float>(t_max) / static_cast<float>(n - 1);
        for (uint32_t i = 0; i < n; ++i) {
            float t = t_max - step * i;
            result.push_back(t_to_sigma(t));
        }
        result.push_back(0);
        return result;
    }
};

struct ExponentialScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        std::vector<float> sigmas;

        // Calculate step size
        float log_sigma_min = std::log(sigma_min);
        float log_sigma_max = std::log(sigma_max);
        float step          = (log_sigma_max - log_sigma_min) / (n - 1);

        // Fill sigmas with exponential values
        for (uint32_t i = 0; i < n; ++i) {
            float sigma = std::exp(log_sigma_max - step * i);
            sigmas.push_back(sigma);
        }

        sigmas.push_back(0.0f);

        return sigmas;
    }
};

/* interp and linear_interp adapted from dpilger26's NumCpp library:
 * https://github.com/dpilger26/NumCpp/tree/5e40aab74d14e257d65d3dc385c9ff9e2120c60e */
constexpr double interp(double left, double right, double perc) noexcept {
    return (left * (1. - perc)) + (right * perc);
}

/* This will make the assumption that the reference x and y values are
 * already sorted in ascending order because they are being generated as
 * such in the calling function */
inline std::vector<double> linear_interp(std::vector<float> new_x,
                                         const std::vector<float> ref_x,
                                         const std::vector<float> ref_y) {
    const size_t len_x = new_x.size();
    size_t i           = 0;
    size_t j           = 0;
    std::vector<double> new_y(len_x);

    if (ref_x.size() != ref_y.size()) {
        LOG_ERROR("Linear Interpolation Failed: length mismatch");
        return new_y;
    }

    /* Adjusted bounds checking to ensure new_x is within ref_x range */
    if (new_x[0] < ref_x[0]) {
        new_x[0] = ref_x[0];
    }
    if (new_x.back() > ref_x.back()) {
        new_x.back() = ref_x.back();
    }

    while (i < len_x) {
        if ((ref_x[j] > new_x[i]) || (new_x[i] > ref_x[j + 1])) {
            j++;
            continue;
        }

        const double perc = static_cast<double>(new_x[i] - ref_x[j]) / static_cast<double>(ref_x[j + 1] - ref_x[j]);

        new_y[i] = interp(ref_y[j], ref_y[j + 1], perc);
        i++;
    }

    return new_y;
}

inline std::vector<float> linear_space(const float start, const float end, const size_t num_points) {
    std::vector<float> result(num_points);
    const float inc = (end - start) / (static_cast<float>(num_points - 1));

    if (num_points > 0) {
        result[0] = start;

        for (size_t i = 1; i < num_points; i++) {
            result[i] = result[i - 1] + inc;
        }
    }

    return result;
}

inline std::vector<float> log_linear_interpolation(std::vector<float> sigma_in,
                                                   const size_t new_len) {
    const size_t s_len        = sigma_in.size();
    std::vector<float> x_vals = linear_space(0.f, 1.f, s_len);
    std::vector<float> y_vals(s_len);

    /* Reverses the input array to be ascending instead of descending,
     * also hits it with a log, it is log-linear interpolation after all */
    for (size_t i = 0; i < s_len; i++) {
        y_vals[i] = std::log(sigma_in[s_len - i - 1]);
    }

    std::vector<float> new_x_vals  = linear_space(0.f, 1.f, new_len);
    std::vector<double> new_y_vals = linear_interp(new_x_vals, x_vals, y_vals);
    std::vector<float> results(new_len);

    for (size_t i = 0; i < new_len; i++) {
        results[i] = static_cast<float>(std::exp(new_y_vals[new_len - i - 1]));
    }

    return results;
}

/*
https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html
*/
struct AYSScheduler : SigmaScheduler {
    SDVersion version;
    explicit AYSScheduler(SDVersion version)
        : version(version) {}
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        const std::vector<float> noise_levels[] = {
            /* SD1.5 */
            {14.6146412293f, 6.4745760956f, 3.8636745985f, 2.6946151520f,
             1.8841921177f, 1.3943805092f, 0.9642583904f, 0.6523686016f,
             0.3977456272f, 0.1515232662f, 0.0291671582f},
            /* SDXL */
            {14.6146412293f, 6.3184485287f, 3.7681790315f, 2.1811480769f,
             1.3405244945f, 0.8620721141f, 0.5550693289f, 0.3798540708f,
             0.2332364134f, 0.1114188177f, 0.0291671582f},
            /* SVD */
            {700.00f, 54.5f, 15.886f, 7.977f, 4.248f, 1.789f, 0.981f, 0.403f,
             0.173f, 0.034f, 0.002f},
        };

        std::vector<float> inputs;
        std::vector<float> results(n + 1);

        if (sd_version_is_sd2((SDVersion)version)) {
            LOG_WARN("AYS_SCHEDULER not designed for SD2.X models");
        } /* fallthrough */
        else if (sd_version_is_sd1((SDVersion)version)) {
            LOG_INFO("AYS_SCHEDULER using SD1.5 noise levels");
            inputs = noise_levels[0];
        } else if (sd_version_is_sdxl((SDVersion)version)) {
            LOG_INFO("AYS_SCHEDULER using SDXL noise levels");
            inputs = noise_levels[1];
        } else if (version == VERSION_SVD) {
            LOG_INFO("AYS_SCHEDULER using SVD noise levels");
            inputs = noise_levels[2];
        } else {
            LOG_ERROR("Version not compatible with AYS_SCHEDULER scheduler");
            return results;
        }

        /* Stretches those pre-calculated reference levels out to the desired
         * size using log-linear interpolation */
        if ((n + 1) != inputs.size()) {
            results = log_linear_interpolation(inputs, n + 1);
        } else {
            results = inputs;
        }

        /* Not sure if this is strictly neccessary */
        results[n] = 0.0f;

        return results;
    }
};

/*
 * GITS Scheduler: https://github.com/zju-pi/diff-sampler/tree/main/gits-main
 */
struct GITSScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        if (sigma_max <= 0.0f) {
            return std::vector<float>{};
        }

        std::vector<float> sigmas;

        // Assume coeff is provided (replace 1.20 with your dynamic coeff)
        float coeff = 1.20f;  // Default coefficient
        // Normalize coeff to the closest value in the array (0.80 to 1.50)
        coeff = std::round(coeff * 20.0f) / 20.0f;  // Round to the nearest 0.05
        // Calculate the index based on the coefficient
        int index = static_cast<int>((coeff - 0.80f) / 0.05f);
        // Ensure the index is within bounds
        index                                                 = std::max(0, std::min(index, static_cast<int>(GITS_NOISE.size() - 1)));
        const std::vector<std::vector<float>>& selected_noise = *GITS_NOISE[index];

        if (n <= 20) {
            sigmas = (selected_noise)[n - 2];
        } else {
            sigmas = log_linear_interpolation(selected_noise.back(), n + 1);
        }

        sigmas[n] = 0.0f;
        return sigmas;
    }
};

struct SGMUniformScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min_in, float sigma_max_in, t_to_sigma_t t_to_sigma_func) override {
        std::vector<float> result;
        if (n == 0) {
            result.push_back(0.0f);
            return result;
        }
        result.reserve(n + 1);
        int t_max                    = TIMESTEPS - 1;
        int t_min                    = 0;
        std::vector<float> timesteps = linear_space(static_cast<float>(t_max), static_cast<float>(t_min), n + 1);
        for (uint32_t i = 0; i < n; i++) {
            result.push_back(t_to_sigma_func(timesteps[i]));
        }
        result.push_back(0.0f);
        return result;
    }
};

struct LCMScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        std::vector<float> result;
        result.reserve(n + 1);
        const int original_steps = 50;
        const int k              = TIMESTEPS / original_steps;
        for (uint32_t i = 0; i < n; i++) {
            // the rounding ensures we match the training schedule of the LCM model
            int index    = (i * original_steps) / n;
            int timestep = (original_steps - index) * k - 1;
            result.push_back(t_to_sigma(static_cast<float>(timestep)));
        }
        result.push_back(0.0f);
        return result;
    }
};

struct KarrasScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        // These *COULD* be function arguments here,
        // but does anybody ever bother to touch them?
        float rho = 7.f;

        if (sigma_min <= 1e-6f) {
            sigma_min = 1e-6f;
        }

        std::vector<float> result(n + 1);

        float min_inv_rho = pow(sigma_min, (1.f / rho));
        float max_inv_rho = pow(sigma_max, (1.f / rho));
        for (uint32_t i = 0; i < n; i++) {
            // Eq. (5) from Karras et al 2022
            result[i] = pow(max_inv_rho + (float)i / ((float)n - 1.f) * (min_inv_rho - max_inv_rho), rho);
        }
        result[n] = 0.;
        return result;
    }
};

struct SimpleScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        std::vector<float> result_sigmas;

        if (n == 0) {
            return result_sigmas;
        }

        result_sigmas.reserve(n + 1);

        int model_sigmas_len = TIMESTEPS;

        float step_factor = static_cast<float>(model_sigmas_len) / static_cast<float>(n);

        for (uint32_t i = 0; i < n; ++i) {
            int offset_from_start_of_py_array = static_cast<int>(static_cast<float>(i) * step_factor);
            int timestep_index                = model_sigmas_len - 1 - offset_from_start_of_py_array;

            if (timestep_index < 0) {
                timestep_index = 0;
            }

            result_sigmas.push_back(t_to_sigma(static_cast<float>(timestep_index)));
        }
        result_sigmas.push_back(0.0f);
        return result_sigmas;
    }
};

// Close to Beta Scheduler, but increadably simple in code.
struct SmoothStepScheduler : SigmaScheduler {
    static constexpr float smoothstep(float x) {
        return x * x * (3.0f - 2.0f * x);
    }

    std::vector<float> get_sigmas(uint32_t n, float /*sigma_min*/, float /*sigma_max*/, t_to_sigma_t t_to_sigma) override {
        std::vector<float> result;
        result.reserve(n + 1);

        const int t_max = TIMESTEPS - 1;
        if (n == 0) {
            return result;
        } else if (n == 1) {
            result.push_back(t_to_sigma((float)t_max));
            result.push_back(0.f);
            return result;
        }

        for (uint32_t i = 0; i < n; i++) {
            float u = 1.f - float(i) / float(n);
            result.push_back(t_to_sigma(std::round(smoothstep(u) * t_max)));
        }

        result.push_back(0.f);
        return result;
    }
};

struct BetaScheduler : SigmaScheduler {
    static double beta_continued_fraction(double a, double b, double x) {
        constexpr int max_iter = 200;
        constexpr double eps = 3.0e-14;
        constexpr double fpmin = 1.0e-300;

        double qab = a + b;
        double qap = a + 1.0;
        double qam = a - 1.0;
        double c = 1.0;
        double d = 1.0 - qab * x / qap;
        if (std::fabs(d) < fpmin) {
            d = fpmin;
        }
        d = 1.0 / d;
        double h = d;

        for (int m = 1; m <= max_iter; ++m) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (std::fabs(d) < fpmin) {
                d = fpmin;
            }
            c = 1.0 + aa / c;
            if (std::fabs(c) < fpmin) {
                c = fpmin;
            }
            d = 1.0 / d;
            h *= d * c;

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (std::fabs(d) < fpmin) {
                d = fpmin;
            }
            c = 1.0 + aa / c;
            if (std::fabs(c) < fpmin) {
                c = fpmin;
            }
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (std::fabs(del - 1.0) <= eps) {
                break;
            }
        }
        return h;
    }

    static double regularized_incomplete_beta(double a, double b, double x) {
        if (x <= 0.0) {
            return 0.0;
        }
        if (x >= 1.0) {
            return 1.0;
        }

        double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                             a * std::log(x) + b * std::log1p(-x));
        if (x < (a + 1.0) / (a + b + 2.0)) {
            return bt * beta_continued_fraction(a, b, x) / a;
        }
        return 1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b;
    }

    static double beta_ppf(double p, double a = 0.6, double b = 0.6) {
        if (p <= 0.0) {
            return 0.0;
        }
        if (p >= 1.0) {
            return 1.0;
        }

        double lo = 0.0;
        double hi = 1.0;
        for (int i = 0; i < 80; ++i) {
            double mid = (lo + hi) * 0.5;
            double cdf = regularized_incomplete_beta(a, b, mid);
            if (cdf < p) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return (lo + hi) * 0.5;
    }

    std::vector<float> get_sigmas(uint32_t n, float /*sigma_min*/, float /*sigma_max*/, t_to_sigma_t t_to_sigma) override {
        std::vector<float> result;
        if (n == 0) {
            return result;
        }

        result.reserve(n + 1);
        const int total_timesteps = TIMESTEPS - 1;
        int last_t = -1;
        for (uint32_t i = 0; i < n; ++i) {
            double p = 1.0 - static_cast<double>(i) / static_cast<double>(n);
            int t = static_cast<int>(std::llround(beta_ppf(p) * total_timesteps));
            t = std::max(0, std::min(total_timesteps, t));
            if (t != last_t) {
                result.push_back(t_to_sigma(static_cast<float>(t)));
            }
            last_t = t;
        }
        result.push_back(0.0f);
        return result;
    }
};

struct BongTangentScheduler : SigmaScheduler {
    static constexpr float kPi = 3.14159265358979323846f;

    static std::vector<float> get_bong_tangent_sigmas(int steps, float slope, float pivot, float start, float end) {
        std::vector<float> sigmas;
        if (steps <= 0) {
            return sigmas;
        }

        float smax   = ((2.0f / kPi) * atanf(-slope * (0.0f - pivot)) + 1.0f) * 0.5f;
        float smin   = ((2.0f / kPi) * atanf(-slope * ((float)(steps - 1) - pivot)) + 1.0f) * 0.5f;
        float srange = smax - smin;
        float sscale = start - end;

        sigmas.reserve(steps);

        if (fabsf(srange) < 1e-8f) {
            if (steps == 1) {
                sigmas.push_back(start);
                return sigmas;
            }
            for (int i = 0; i < steps; ++i) {
                float t = (float)i / (float)(steps - 1);
                sigmas.push_back(start + (end - start) * t);
            }
            return sigmas;
        }

        float inv_srange = 1.0f / srange;
        for (int x = 0; x < steps; ++x) {
            float v     = ((2.0f / kPi) * atanf(-slope * ((float)x - pivot)) + 1.0f) * 0.5f;
            float sigma = ((v - smin) * inv_srange) * sscale + end;
            sigmas.push_back(sigma);
        }

        return sigmas;
    }

    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t /*t_to_sigma*/) override {
        std::vector<float> result;
        if (n == 0) {
            return result;
        }

        float start  = sigma_max;
        float end    = sigma_min;
        float middle = sigma_min + (sigma_max - sigma_min) * 0.5f;

        float pivot_1 = 0.6f;
        float pivot_2 = 0.6f;
        float slope_1 = 0.2f;
        float slope_2 = 0.2f;

        int steps     = static_cast<int>(n) + 2;
        int midpoint  = static_cast<int>(((float)steps * pivot_1 + (float)steps * pivot_2) * 0.5f);
        int pivot_1_i = static_cast<int>((float)steps * pivot_1);
        int pivot_2_i = static_cast<int>((float)steps * pivot_2);

        float slope_scale = (float)steps / 40.0f;
        slope_1           = slope_1 / slope_scale;
        slope_2           = slope_2 / slope_scale;

        int stage_2_len = steps - midpoint;
        int stage_1_len = steps - stage_2_len;

        std::vector<float> sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1, (float)pivot_1_i, start, middle);
        std::vector<float> sigmas_2 = get_bong_tangent_sigmas(stage_2_len, slope_2, (float)(pivot_2_i - stage_1_len), middle, end);

        if (!sigmas_1.empty()) {
            sigmas_1.pop_back();
        }

        result.reserve(n + 1);
        result.insert(result.end(), sigmas_1.begin(), sigmas_1.end());
        result.insert(result.end(), sigmas_2.begin(), sigmas_2.end());

        if (result.size() < n + 1) {
            while (result.size() < n + 1) {
                result.push_back(end);
            }
        } else if (result.size() > n + 1) {
            result.resize(n + 1);
        }

        result[n] = 0.0f;
        return result;
    }
};

struct KLOptimalScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        std::vector<float> sigmas;

        if (n == 0) {
            return sigmas;
        }

        if (n == 1) {
            sigmas.push_back(sigma_max);
            sigmas.push_back(0.0f);
            return sigmas;
        }

        if (sigma_min <= 1e-6f) {
            sigma_min = 1e-6f;
        }

        sigmas.reserve(n + 1);

        float alpha_min = std::atan(sigma_min);
        float alpha_max = std::atan(sigma_max);

        for (uint32_t i = 0; i < n; ++i) {
            float t     = static_cast<float>(i) / static_cast<float>(n - 1);
            float angle = t * alpha_min + (1.0f - t) * alpha_max;
            sigmas.push_back(std::tan(angle));
        }

        sigmas.push_back(0.0f);

        return sigmas;
    }
};

struct Denoiser {
    virtual float sigma_min()                                                        = 0;
    virtual float sigma_max()                                                        = 0;
    virtual float sigma_to_t(float sigma)                                            = 0;
    virtual float t_to_sigma(float t)                                                = 0;
    virtual std::vector<float> get_scalings(float sigma)                             = 0;
    virtual sd::Tensor<float> noise_scaling(float sigma,
                                            const sd::Tensor<float>& noise,
                                            const sd::Tensor<float>& latent)         = 0;
    virtual sd::Tensor<float> inverse_noise_scaling(float sigma,
                                                    const sd::Tensor<float>& latent) = 0;

    virtual std::vector<float> get_sigmas(uint32_t n, int /*image_seq_len*/, scheduler_t scheduler_type, SDVersion version) {
        auto bound_t_to_sigma = std::bind(&Denoiser::t_to_sigma, this, std::placeholders::_1);
        std::shared_ptr<SigmaScheduler> scheduler;
        switch (scheduler_type) {
            case DISCRETE_SCHEDULER:
                LOG_INFO("get_sigmas with discrete scheduler");
                scheduler = std::make_shared<DiscreteScheduler>();
                break;
            case KARRAS_SCHEDULER:
                LOG_INFO("get_sigmas with Karras scheduler");
                scheduler = std::make_shared<KarrasScheduler>();
                break;
            case EXPONENTIAL_SCHEDULER:
                LOG_INFO("get_sigmas exponential scheduler");
                scheduler = std::make_shared<ExponentialScheduler>();
                break;
            case AYS_SCHEDULER:
                LOG_INFO("get_sigmas with Align-Your-Steps scheduler");
                scheduler = std::make_shared<AYSScheduler>(version);
                break;
            case GITS_SCHEDULER:
                LOG_INFO("get_sigmas with GITS scheduler");
                scheduler = std::make_shared<GITSScheduler>();
                break;
            case SGM_UNIFORM_SCHEDULER:
                LOG_INFO("get_sigmas with SGM Uniform scheduler");
                scheduler = std::make_shared<SGMUniformScheduler>();
                break;
            case SIMPLE_SCHEDULER:
                LOG_INFO("get_sigmas with Simple scheduler");
                scheduler = std::make_shared<SimpleScheduler>();
                break;
            case SMOOTHSTEP_SCHEDULER:
                LOG_INFO("get_sigmas with SmoothStep scheduler");
                scheduler = std::make_shared<SmoothStepScheduler>();
                break;
            case BONG_TANGENT_SCHEDULER:
                LOG_INFO("get_sigmas with bong_tangent scheduler");
                scheduler = std::make_shared<BongTangentScheduler>();
                break;
            case BETA_SCHEDULER:
                LOG_INFO("get_sigmas with beta scheduler");
                scheduler = std::make_shared<BetaScheduler>();
                break;
            case KL_OPTIMAL_SCHEDULER:
                LOG_INFO("get_sigmas with KL Optimal scheduler");
                scheduler = std::make_shared<KLOptimalScheduler>();
                break;
            case LCM_SCHEDULER:
                LOG_INFO("get_sigmas with LCM scheduler");
                scheduler = std::make_shared<LCMScheduler>();
                break;
            default:
                LOG_INFO("get_sigmas with discrete scheduler (default)");
                scheduler = std::make_shared<DiscreteScheduler>();
                break;
        }
        return scheduler->get_sigmas(n, sigma_min(), sigma_max(), bound_t_to_sigma);
    }
};

struct CompVisDenoiser : public Denoiser {
    float sigmas[TIMESTEPS];
    float log_sigmas[TIMESTEPS];

    float sigma_data = 1.0f;

    float sigma_min() override {
        return sigmas[0];
    }

    float sigma_max() override {
        return sigmas[TIMESTEPS - 1];
    }

    float sigma_to_t(float sigma) override {
        float log_sigma = std::log(sigma);
        std::vector<float> dists;
        dists.reserve(TIMESTEPS);
        for (float log_sigma_val : log_sigmas) {
            dists.push_back(log_sigma - log_sigma_val);
        }

        int low_idx = 0;
        for (size_t i = 0; i < TIMESTEPS; i++) {
            if (dists[i] >= 0) {
                low_idx++;
            }
        }
        low_idx      = std::min(std::max(low_idx - 1, 0), TIMESTEPS - 2);
        int high_idx = low_idx + 1;

        float low  = log_sigmas[low_idx];
        float high = log_sigmas[high_idx];
        float w    = (low - log_sigma) / (low - high);
        w          = std::max(0.f, std::min(1.f, w));
        float t    = (1.0f - w) * low_idx + w * high_idx;

        return t;
    }

    float t_to_sigma(float t) override {
        int low_idx     = static_cast<int>(std::floor(t));
        int high_idx    = static_cast<int>(std::ceil(t));
        float w         = t - static_cast<float>(low_idx);
        float log_sigma = (1.0f - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx];
        return std::exp(log_sigma);
    }

    std::vector<float> get_scalings(float sigma) override {
        float c_skip = 1.0f;
        float c_out  = -sigma;
        float c_in   = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_skip, c_out, c_in};
    }

    virtual sd::Tensor<float> noise_scaling(float sigma,
                                            const sd::Tensor<float>& noise,
                                            const sd::Tensor<float>& latent) override {
        GGML_ASSERT(noise.numel() == latent.numel());
        return latent + noise * sigma;
    }

    sd::Tensor<float> inverse_noise_scaling(float sigma, const sd::Tensor<float>& latent) override {
        SD_UNUSED(sigma);
        return latent;
    }
};

struct CompVisVDenoiser : public CompVisDenoiser {
    std::vector<float> get_scalings(float sigma) override {
        float c_skip = sigma_data * sigma_data / (sigma * sigma + sigma_data * sigma_data);
        float c_out  = -sigma * sigma_data / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        float c_in   = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_skip, c_out, c_in};
    }
};

struct EDMVDenoiser : public CompVisVDenoiser {
    float min_sigma = 0.002f;
    float max_sigma = 120.0f;

    EDMVDenoiser(float min_sigma = 0.002, float max_sigma = 120.0)
        : min_sigma(min_sigma), max_sigma(max_sigma) {
    }

    float t_to_sigma(float t) override {
        return std::exp(t * 4 / (float)TIMESTEPS);
    }

    float sigma_to_t(float s) override {
        return 0.25f * std::log(s);
    }

    float sigma_min() override {
        return min_sigma;
    }

    float sigma_max() override {
        return max_sigma;
    }
};

inline float time_snr_shift(float alpha, float t) {
    if (alpha == 1.0f) {
        return t;
    }
    return alpha * t / (1 + (alpha - 1) * t);
}

struct DiscreteFlowDenoiser : public Denoiser {
    float shift = 3.0f;
    bool explicit_shift = false;

    DiscreteFlowDenoiser(float shift = 3.0f) {
        this->shift = shift;
    }

    void set_shift(float shift, bool is_explicit = true) {
        this->shift = shift;
        explicit_shift = is_explicit;
    }

    bool has_explicit_shift() const {
        return explicit_shift;
    }

    float sigma_min() override {
        return t_to_sigma(0);
    }

    float sigma_max() override {
        return t_to_sigma(TIMESTEPS - 1);
    }

    float sigma_to_t(float sigma) override {
        return sigma * 1000.f;
    }

    float t_to_sigma(float t) override {
        t = t + 1;
        return time_snr_shift(shift, t / 1000.f);
    }

    std::vector<float> get_scalings(float sigma) override {
        float c_skip = 1.0f;
        float c_out  = -sigma;
        float c_in   = 1.0f;
        return {c_skip, c_out, c_in};
    }

    sd::Tensor<float> noise_scaling(float sigma,
                                    const sd::Tensor<float>& noise,
                                    const sd::Tensor<float>& latent) override {
        GGML_ASSERT(noise.numel() == latent.numel());
        return latent * (1.0f - sigma) + noise * sigma;
    }
    sd::Tensor<float> inverse_noise_scaling(float sigma, const sd::Tensor<float>& latent) override {
        return latent * (1.0f / (1.0f - sigma));
    }
};

inline float flux_time_shift(float mu, float sigma, float t) {
    return ::expf(mu) / (::expf(mu) + ::powf((1.0f / t - 1.0f), sigma));
}

struct FluxFlowDenoiser : public DiscreteFlowDenoiser {
    FluxFlowDenoiser() = default;

    float sigma_to_t(float sigma) override {
        return sigma;
    }

    float t_to_sigma(float t) override {
        t = t + 1;
        return flux_time_shift(shift, 1.0f, t / TIMESTEPS);
    }
};

struct Flux2FlowDenoiser : public FluxFlowDenoiser {
    Flux2FlowDenoiser() = default;

    float t_to_sigma(float t) override {
        t = t + 1;
        const float normalized = t / TIMESTEPS;
        if (has_explicit_shift()) {
            return time_snr_shift(shift, normalized);
        }
        return flux_time_shift(shift, 1.0f, normalized);
    }

    float compute_empirical_mu(uint32_t n, int image_seq_len) {
        const float a1 = 8.73809524e-05f;
        const float b1 = 1.89833333f;
        const float a2 = 0.00016927f;
        const float b2 = 0.45666666f;

        if (image_seq_len > 4300) {
            float mu = a2 * image_seq_len + b2;
            return mu;
        }

        float m_200 = a2 * image_seq_len + b2;
        float m_10  = a1 * image_seq_len + b1;

        float a  = (m_200 - m_10) / 190.0f;
        float b  = m_200 - 200.0f * a;
        float mu = a * n + b;

        return mu;
    }

    std::vector<float> get_sigmas(uint32_t n, int image_seq_len, scheduler_t scheduler_type, SDVersion version) override {
        const char* bonsai_int1 = std::getenv("SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1");
        const bool bonsai_dynamic_mu = bonsai_int1 != nullptr && bonsai_int1[0] != '\0' && bonsai_int1[0] != '0';
        if (bonsai_dynamic_mu && scheduler_type == DISCRETE_SCHEDULER) {
            const float mu = compute_empirical_mu(n, image_seq_len);
            LOG_INFO("get_sigmas with Bonsai Prism-compatible Flux2 FlowMatchEuler dynamic mu %.6f", mu);
            std::vector<float> result;
            if (n == 0) {
                return result;
            }
            result.reserve(static_cast<size_t>(n) + 1);
            for (uint32_t i = 0; i < n; ++i) {
                const float base_sigma = 1.0f - static_cast<float>(i) * (1.0f - 1.0f / static_cast<float>(n)) / static_cast<float>(std::max<uint32_t>(1, n - 1));
                result.push_back(flux_time_shift(mu, 1.0f, base_sigma));
            }
            result.push_back(0.0f);
            return result;
        }
        if (!has_explicit_shift()) {
            float mu = compute_empirical_mu(n, image_seq_len);
            LOG_DEBUG("Flux2FlowDenoiser: set shift to %.3f", mu);
            set_shift(mu, false);
        } else {
            LOG_DEBUG("Flux2FlowDenoiser: using explicit shift %.3f", shift);
        }
        if (has_explicit_shift() && scheduler_type == DISCRETE_SCHEDULER) {
            // Match Diffusers FlowMatchEulerDiscreteScheduler's default inference schedule:
            // linearly interpolate between sigma_max and sigma_min in sigma-to-t space,
            // convert back to normalized sigma, then apply the fixed shift again.
            LOG_INFO("get_sigmas with Flux2 FlowMatchEulerDiscreteScheduler-compatible discrete scheduler");
            std::vector<float> result;
            if (n == 0) {
                return result;
            }
            result.reserve(static_cast<size_t>(n) + 1);

            const float max_t = sigma_max() * static_cast<float>(TIMESTEPS);
            const float min_t = sigma_min() * static_cast<float>(TIMESTEPS);
            if (n == 1) {
                result.push_back(time_snr_shift(shift, max_t / static_cast<float>(TIMESTEPS)));
                result.push_back(0.0f);
                return result;
            }
            const float step = (max_t - min_t) / static_cast<float>(n - 1);
            for (uint32_t i = 0; i < n; ++i) {
                const float timestep = max_t - step * static_cast<float>(i);
                const float sigma = timestep / static_cast<float>(TIMESTEPS);
                result.push_back(time_snr_shift(shift, sigma));
            }
            result.push_back(0.0f);
            return result;
        }
        return Denoiser::get_sigmas(n, image_seq_len, scheduler_type, version);
    }
};

typedef std::function<sd::Tensor<float>(const sd::Tensor<float>&, float, int)> denoise_cb_t;

static std::pair<float, float> get_ancestral_step(float sigma_from,
                                                  float sigma_to,
                                                  float eta = 1.0f) {
    float sigma_up   = 0.0f;
    float sigma_down = sigma_to;

    if (eta <= 0.0f) {
        return {sigma_down, sigma_up};
    }

    float sigma_from_sq = sigma_from * sigma_from;
    float sigma_to_sq   = sigma_to * sigma_to;
    if (sigma_from_sq > 0.0f) {
        float term = sigma_to_sq * (sigma_from_sq - sigma_to_sq) / sigma_from_sq;
        sigma_up   = std::min(sigma_to, eta * std::sqrt(std::max(term, 0.0f)));
    }

    float sigma_down_sq = sigma_to_sq - sigma_up * sigma_up;
    sigma_down          = sigma_down_sq > 0.0f ? std::sqrt(sigma_down_sq) : 0.0f;
    return {sigma_down, sigma_up};
}

static std::tuple<float, float, float> get_ancestral_step_flow(float sigma_from,
                                                               float sigma_to,
                                                               float eta = 1.0f) {
    float sigma_down  = sigma_to;
    float sigma_up    = 0.0f;
    float alpha_scale = 1.0f;

    if (eta <= 0.0f || sigma_from <= 0.0f || sigma_to <= 0.0f) {
        return {sigma_down, sigma_up, alpha_scale};
    }

    // Flow Euler ancestral sampling becomes numerically unstable for eta > 1, so
    // clamp to the valid maximum-noise regime instead of letting NaNs propagate.
    eta = std::min(eta, 1.0f);

    float sigma_ratio = sigma_to / sigma_from;
    sigma_down        = sigma_to * (1.0f + (sigma_ratio - 1.0f) * eta);
    sigma_down        = std::max(0.0f, std::min(sigma_to, sigma_down));

    float denom = 1.0f - sigma_down;
    if (denom <= 0.0f) {
        return {sigma_to, sigma_up, alpha_scale};
    }

    alpha_scale = (1.0f - sigma_to) / denom;

    float term = (sigma_down / sigma_to) * alpha_scale;
    term       = std::max(-1.0f, std::min(1.0f, term));
    sigma_up   = sigma_to * std::sqrt(std::max(1.0f - term * term, 0.0f));
    return {sigma_down, sigma_up, alpha_scale};
}

static std::tuple<float, float, float> get_ancestral_step(float sigma_from,
                                                          float sigma_to,
                                                          float eta,
                                                          bool is_flow_denoiser) {
    if (is_flow_denoiser) {
        return get_ancestral_step_flow(sigma_from, sigma_to, eta);
    }

    auto [sigma_down, sigma_up] = get_ancestral_step(sigma_from, sigma_to, eta);
    return {sigma_down, sigma_up, 1.0f};
}

static sd::Tensor<float> sample_euler_ancestral(denoise_cb_t model,
                                                sd::Tensor<float> x,
                                                const std::vector<float>& sigmas,
                                                std::shared_ptr<RNG> rng,
                                                float eta) {
    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        float sigma       = sigmas[i];
        auto denoised_opt = model(x, sigma, i + 1);
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised  = std::move(denoised_opt);
        sd::Tensor<float> d         = (x - denoised) / sigma;
        auto [sigma_down, sigma_up] = get_ancestral_step(sigmas[i], sigmas[i + 1], eta);
        x += d * (sigma_down - sigmas[i]);
        if (sigmas[i + 1] > 0) {
            x += sd::Tensor<float>::randn_like(x, rng) * sigma_up;
        }
    }
    return x;
}

static sd::Tensor<float> sample_euler_flow(denoise_cb_t model,
                                           sd::Tensor<float> x,
                                           const std::vector<float>& sigmas,
                                           std::shared_ptr<RNG> rng,
                                           float eta) {
    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        float sigma       = sigmas[i];
        auto denoised_opt = model(x, sigma, i + 1);
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised               = std::move(denoised_opt);
        auto [sigma_down, sigma_up, alpha_scale] = get_ancestral_step_flow(sigma, sigmas[i + 1], eta);
        float sigma_ratio                        = sigma_down / sigma;
        x                                        = sigma_ratio * x + (1.0f - sigma_ratio) * denoised;

        if (sigma_up > 0.0f) {
            x = alpha_scale * x + sd::Tensor<float>::randn_like(x, rng) * sigma_up;
        }
    }
    return x;
}

static sd::Tensor<float> sample_euler(denoise_cb_t model,
                                      sd::Tensor<float> x,
                                      const std::vector<float>& sigmas) {
    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        float sigma       = sigmas[i];
        auto denoised_opt = model(x, sigma, i + 1);
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised = std::move(denoised_opt);
        sd::Tensor<float> d        = (x - denoised) / sigma;
        x += d * (sigmas[i + 1] - sigma);
    }
    return x;
}

static sd::Tensor<float> sample_heun(denoise_cb_t model,
                                     sd::Tensor<float> x,
                                     const std::vector<float>& sigmas) {
    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        auto denoised_opt = model(x, sigmas[i], -(i + 1));
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised = std::move(denoised_opt);
        sd::Tensor<float> d        = (x - denoised) / sigmas[i];
        float dt                   = sigmas[i + 1] - sigmas[i];
        if (sigmas[i + 1] == 0) {
            x += d * dt;
        } else {
            sd::Tensor<float> x2 = x + d * dt;
            auto denoised2_opt   = model(x2, sigmas[i + 1], i + 1);
            if (denoised2_opt.empty()) {
                return {};
            }
            sd::Tensor<float> denoised2 = std::move(denoised2_opt);
            d                           = (d + (x2 - denoised2) / sigmas[i + 1]) / 2.0f;
            x += d * dt;
        }
    }
    return x;
}

static sd::Tensor<float> sample_dpm2(denoise_cb_t model,
                                     sd::Tensor<float> x,
                                     const std::vector<float>& sigmas) {
    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        auto denoised_opt = model(x, sigmas[i], -(i + 1));
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised = std::move(denoised_opt);
        sd::Tensor<float> d        = (x - denoised) / sigmas[i];
        if (sigmas[i + 1] == 0) {
            x += d * (sigmas[i + 1] - sigmas[i]);
        } else {
            float sigma_mid      = exp(0.5f * (log(sigmas[i]) + log(sigmas[i + 1])));
            float dt_1           = sigma_mid - sigmas[i];
            float dt_2           = sigmas[i + 1] - sigmas[i];
            sd::Tensor<float> x2 = x + d * dt_1;
            auto denoised2_opt   = model(x2, sigma_mid, i + 1);
            if (denoised2_opt.empty()) {
                return {};
            }
            sd::Tensor<float> denoised2 = std::move(denoised2_opt);
            x += ((x2 - denoised2) / sigma_mid) * dt_2;
        }
    }
    return x;
}

static sd::Tensor<float> sample_dpmpp_2s_ancestral(denoise_cb_t model,
                                                   sd::Tensor<float> x,
                                                   const std::vector<float>& sigmas,
                                                   std::shared_ptr<RNG> rng,
                                                   float eta) {
    auto t_fn     = [](float sigma) -> float { return -log(sigma); };
    auto sigma_fn = [](float t) -> float { return exp(-t); };

    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        auto denoised_opt = model(x, sigmas[i], -(i + 1));
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised  = std::move(denoised_opt);
        auto [sigma_down, sigma_up] = get_ancestral_step(sigmas[i], sigmas[i + 1], eta);

        if (sigma_down == 0) {
            x = denoised;
        } else {
            float t              = t_fn(sigmas[i]);
            float t_next         = t_fn(sigma_down);
            float h              = t_next - t;
            float s              = t + 0.5f * h;
            sd::Tensor<float> x2 = (sigma_fn(s) / sigma_fn(t)) * x - (exp(-h * 0.5f) - 1) * denoised;
            auto denoised2_opt   = model(x2, sigmas[i + 1], i + 1);
            if (denoised2_opt.empty()) {
                return {};
            }
            sd::Tensor<float> denoised2 = std::move(denoised2_opt);
            x                           = (sigma_fn(t_next) / sigma_fn(t)) * x - (exp(-h) - 1) * denoised2;
        }

        if (sigmas[i + 1] > 0) {
            x += sd::Tensor<float>::randn_like(x, rng) * sigma_up;
        }
    }
    return x;
}

static sd::Tensor<float> sample_dpmpp_2m(denoise_cb_t model,
                                         sd::Tensor<float> x,
                                         const std::vector<float>& sigmas) {
    sd::Tensor<float> old_denoised = x;
    auto t_fn                      = [](float sigma) -> float { return -log(sigma); };

    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        auto denoised_opt = model(x, sigmas[i], i + 1);
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised = std::move(denoised_opt);
        float t                    = t_fn(sigmas[i]);
        float t_next               = t_fn(sigmas[i + 1]);
        float h                    = t_next - t;
        float a                    = sigmas[i + 1] / sigmas[i];
        float b                    = exp(-h) - 1.f;

        if (i == 0 || sigmas[i + 1] == 0) {
            x = a * x - b * denoised;
        } else {
            float h_last                 = t - t_fn(sigmas[i - 1]);
            float r                      = h_last / h;
            sd::Tensor<float> denoised_d = (1.f + 1.f / (2.f * r)) * denoised - (1.f / (2.f * r)) * old_denoised;
            x                            = a * x - b * denoised_d;
        }
        old_denoised = denoised;
    }
    return x;
}

static sd::Tensor<float> sample_dpmpp_2m_v2(denoise_cb_t model,
                                            sd::Tensor<float> x,
                                            const std::vector<float>& sigmas) {
    sd::Tensor<float> old_denoised = x;
    auto t_fn                      = [](float sigma) -> float { return -log(sigma); };

    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        auto denoised_opt = model(x, sigmas[i], i + 1);
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised = std::move(denoised_opt);
        float t                    = t_fn(sigmas[i]);
        float t_next               = t_fn(sigmas[i + 1]);
        float h                    = t_next - t;
        float a                    = sigmas[i + 1] / sigmas[i];

        if (i == 0 || sigmas[i + 1] == 0) {
            float b = exp(-h) - 1.f;
            x       = a * x - b * denoised;
        } else {
            float h_last                 = t - t_fn(sigmas[i - 1]);
            float h_min                  = std::min(h_last, h);
            float h_max                  = std::max(h_last, h);
            float r                      = h_max / h_min;
            float h_d                    = (h_max + h_min) / 2.f;
            float b                      = exp(-h_d) - 1.f;
            sd::Tensor<float> denoised_d = (1.f + 1.f / (2.f * r)) * denoised - (1.f / (2.f * r)) * old_denoised;
            x                            = a * x - b * denoised_d;
        }
        old_denoised = denoised;
    }
    return x;
}

static float dpmpp_clamp_flow_sigma(float sigma) {
    return std::max(1e-6f, std::min(1.0f - 1e-6f, sigma));
}

static float dpmpp_sigma_to_half_log_snr(float sigma, bool is_flow_denoiser = false) {
    if (is_flow_denoiser) {
        sigma = dpmpp_clamp_flow_sigma(sigma);
        return std::log((1.0f - sigma) / sigma);
    }
    return -std::log(std::max(sigma, 1e-20f));
}

static float dpmpp_half_log_snr_to_sigma(float lambda, bool is_flow_denoiser = false) {
    if (is_flow_denoiser) {
        if (lambda > 80.0f) {
            return 0.0f;
        }
        if (lambda < -80.0f) {
            return 1.0f;
        }
        return 1.0f / (1.0f + std::exp(lambda));
    }
    return std::exp(-lambda);
}

static void dpmpp_offset_first_sigma_for_snr(std::vector<float>& sigmas, bool is_flow_denoiser) {
    if (is_flow_denoiser && sigmas.size() > 1 && sigmas[0] >= 1.0f) {
        sigmas[0] = 1.0f - 1e-4f;
    }
}

static float dpmpp_expm1_neg(float x) {
    return std::expm1(-x);
}

static float dpmpp_phi2(float h_eta) {
    if (std::fabs(h_eta) < 1e-6f) {
        return 0.5f * h_eta - (h_eta * h_eta) / 6.0f;
    }
    return dpmpp_expm1_neg(h_eta) / h_eta + 1.0f;
}

static float dpmpp_phi3(float h_eta) {
    if (std::fabs(h_eta) < 1e-6f) {
        return -h_eta / 6.0f + (h_eta * h_eta) / 24.0f;
    }
    return dpmpp_phi2(h_eta) / h_eta - 0.5f;
}

static float dpmpp_sde_noise_scale(float h, float eta, float s_noise, float sigma_next) {
    if (eta <= 0.0f || s_noise <= 0.0f || sigma_next <= 0.0f) {
        return 0.0f;
    }
    const float variance = -std::expm1(-2.0f * h * eta);
    return sigma_next * std::sqrt(std::max(variance, 0.0f)) * s_noise;
}

static sd::Tensor<float> sample_dpmpp_sde(denoise_cb_t model,
                                          sd::Tensor<float> x,
                                          std::vector<float> sigmas,
                                          std::shared_ptr<RNG> rng,
                                          float eta,
                                          float s_noise,
                                          float r,
                                          bool is_flow_denoiser) {
    if (sigmas.size() <= 1) {
        return x;
    }
    if (r <= 0.0f || !std::isfinite(r)) {
        LOG_ERROR("dpmpp_sde r must be greater than 0");
        return {};
    }

    dpmpp_offset_first_sigma_for_snr(sigmas, is_flow_denoiser);
    const int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        auto denoised_opt = model(x, sigmas[i], -(i + 1));
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised = std::move(denoised_opt);

        if (sigmas[i + 1] == 0.0f) {
            x = denoised;
            continue;
        }

        const float lambda_s   = dpmpp_sigma_to_half_log_snr(sigmas[i], is_flow_denoiser);
        const float lambda_t   = dpmpp_sigma_to_half_log_snr(sigmas[i + 1], is_flow_denoiser);
        const float h          = lambda_t - lambda_s;
        const float lambda_s_1 = lambda_s + r * h;
        const float fac        = 1.0f / (2.0f * r);
        const float sigma_s_1  = dpmpp_half_log_snr_to_sigma(lambda_s_1, is_flow_denoiser);
        const float alpha_s    = sigmas[i] * std::exp(lambda_s);
        const float alpha_s_1  = sigma_s_1 * std::exp(lambda_s_1);
        const float alpha_t    = sigmas[i + 1] * std::exp(lambda_t);

        auto [sigma_down_1, sigma_up_1] = get_ancestral_step(std::exp(-lambda_s),
                                                             std::exp(-lambda_s_1),
                                                             eta);
        if (sigma_down_1 <= 0.0f || !std::isfinite(sigma_down_1)) {
            LOG_ERROR("dpmpp_sde produced invalid intermediate sigma");
            return {};
        }
        const float lambda_s_1_adjusted = -std::log(sigma_down_1);
        const float h_adjusted_1        = lambda_s_1_adjusted - lambda_s;
        sd::Tensor<float> x_2           = (alpha_s_1 / alpha_s) * std::exp(-h_adjusted_1) * x -
                                alpha_s_1 * std::expm1(-h_adjusted_1) * denoised;
        if (eta > 0.0f && s_noise > 0.0f && sigma_up_1 > 0.0f) {
            x_2 += alpha_s_1 * sd::Tensor<float>::randn_like(x, rng) * (s_noise * sigma_up_1);
        }

        auto denoised_2_opt = model(x_2, sigma_s_1, i + 1);
        if (denoised_2_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised_2 = std::move(denoised_2_opt);

        auto [sigma_down_t, sigma_up_t] = get_ancestral_step(std::exp(-lambda_s),
                                                             std::exp(-lambda_t),
                                                             eta);
        if (sigma_down_t <= 0.0f || !std::isfinite(sigma_down_t)) {
            LOG_ERROR("dpmpp_sde produced invalid target sigma");
            return {};
        }
        const float lambda_t_adjusted = -std::log(sigma_down_t);
        const float h_adjusted_t      = lambda_t_adjusted - lambda_s;
        sd::Tensor<float> denoised_d  = (1.0f - fac) * denoised + fac * denoised_2;
        x                             = (alpha_t / alpha_s) * std::exp(-h_adjusted_t) * x -
            alpha_t * std::expm1(-h_adjusted_t) * denoised_d;
        if (eta > 0.0f && s_noise > 0.0f && sigma_up_t > 0.0f) {
            x += alpha_t * sd::Tensor<float>::randn_like(x, rng) * (s_noise * sigma_up_t);
        }
    }
    return x;
}

static sd::Tensor<float> sample_dpmpp_2m_sde(denoise_cb_t model,
                                             sd::Tensor<float> x,
                                             std::vector<float> sigmas,
                                             std::shared_ptr<RNG> rng,
                                             float eta,
                                             float s_noise,
                                             dpmpp_sde_solver_t solver_type,
                                             bool is_flow_denoiser) {
    if (solver_type != DPMPP_SDE_SOLVER_MIDPOINT && solver_type != DPMPP_SDE_SOLVER_HEUN) {
        LOG_ERROR("dpmpp_2m_sde solver_type must be midpoint or heun");
        return {};
    }

    sd::Tensor<float> old_denoised;
    bool have_old_denoised = false;
    float h_last           = 0.0f;

    dpmpp_offset_first_sigma_for_snr(sigmas, is_flow_denoiser);
    const int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        auto denoised_opt = model(x, sigmas[i], i + 1);
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised = std::move(denoised_opt);

        if (sigmas[i + 1] == 0.0f) {
            x = denoised;
        } else {
            const float lambda_s = dpmpp_sigma_to_half_log_snr(sigmas[i], is_flow_denoiser);
            const float lambda_t = dpmpp_sigma_to_half_log_snr(sigmas[i + 1], is_flow_denoiser);
            const float h        = lambda_t - lambda_s;
            const float h_eta    = h * (eta + 1.0f);
            const float alpha_t  = sigmas[i + 1] * std::exp(lambda_t);

            x = (sigmas[i + 1] / sigmas[i]) * std::exp(-h * eta) * x +
                alpha_t * (-std::expm1(-h_eta)) * denoised;

            if (have_old_denoised && h != 0.0f && h_last != 0.0f) {
                const float r = h_last / h;
                if (solver_type == DPMPP_SDE_SOLVER_HEUN) {
                    x += alpha_t * dpmpp_phi2(h_eta) * (1.0f / r) * (denoised - old_denoised);
                } else {
                    x += 0.5f * alpha_t * (-std::expm1(-h_eta)) * (1.0f / r) * (denoised - old_denoised);
                }
            }

            const float noise_scale = dpmpp_sde_noise_scale(h, eta, s_noise, sigmas[i + 1]);
            if (noise_scale > 0.0f) {
                x += sd::Tensor<float>::randn_like(x, rng) * noise_scale;
            }

            h_last = h;
        }

        old_denoised      = denoised;
        have_old_denoised = true;
    }
    return x;
}

static sd::Tensor<float> sample_dpmpp_3m_sde(denoise_cb_t model,
                                             sd::Tensor<float> x,
                                             std::vector<float> sigmas,
                                             std::shared_ptr<RNG> rng,
                                             float eta,
                                             float s_noise,
                                             bool is_flow_denoiser) {
    sd::Tensor<float> denoised_1;
    sd::Tensor<float> denoised_2;
    bool have_denoised_1 = false;
    bool have_denoised_2 = false;
    float h_1            = 0.0f;
    float h_2            = 0.0f;

    dpmpp_offset_first_sigma_for_snr(sigmas, is_flow_denoiser);
    const int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        auto denoised_opt = model(x, sigmas[i], i + 1);
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised = std::move(denoised_opt);

        if (sigmas[i + 1] == 0.0f) {
            x = denoised;
        } else {
            const float lambda_s = dpmpp_sigma_to_half_log_snr(sigmas[i], is_flow_denoiser);
            const float lambda_t = dpmpp_sigma_to_half_log_snr(sigmas[i + 1], is_flow_denoiser);
            const float h        = lambda_t - lambda_s;
            const float h_eta    = h * (eta + 1.0f);
            const float alpha_t  = sigmas[i + 1] * std::exp(lambda_t);

            x = (sigmas[i + 1] / sigmas[i]) * std::exp(-h * eta) * x +
                alpha_t * (-std::expm1(-h_eta)) * denoised;

            if (have_denoised_2 && h != 0.0f && h_1 != 0.0f && h_2 != 0.0f) {
                const float r0 = h_1 / h;
                const float r1 = h_2 / h;
                if (r0 + r1 != 0.0f) {
                    sd::Tensor<float> d1_0 = (denoised - denoised_1) / r0;
                    sd::Tensor<float> d1_1 = (denoised_1 - denoised_2) / r1;
                    sd::Tensor<float> d1   = d1_0 + (d1_0 - d1_1) * (r0 / (r0 + r1));
                    sd::Tensor<float> d2   = (d1_0 - d1_1) / (r0 + r1);
                    x += (alpha_t * dpmpp_phi2(h_eta)) * d1 - (alpha_t * dpmpp_phi3(h_eta)) * d2;
                }
            } else if (have_denoised_1 && h != 0.0f && h_1 != 0.0f) {
                const float r            = h_1 / h;
                sd::Tensor<float> d      = (denoised - denoised_1) / r;
                x += (alpha_t * dpmpp_phi2(h_eta)) * d;
            }

            const float noise_scale = dpmpp_sde_noise_scale(h, eta, s_noise, sigmas[i + 1]);
            if (noise_scale > 0.0f) {
                x += sd::Tensor<float>::randn_like(x, rng) * noise_scale;
            }

            h_2 = h_1;
            h_1 = h;
        }

        denoised_2      = denoised_1;
        denoised_1      = denoised;
        have_denoised_2 = have_denoised_1;
        have_denoised_1 = true;
    }
    return x;
}

static sd::Tensor<float> sample_lcm(denoise_cb_t model,
                                    sd::Tensor<float> x,
                                    const std::vector<float>& sigmas,
                                    std::shared_ptr<RNG> rng) {
    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        auto denoised_opt = model(x, sigmas[i], i + 1);
        if (denoised_opt.empty()) {
            return {};
        }
        x = std::move(denoised_opt);
        if (sigmas[i + 1] > 0) {
            x += sd::Tensor<float>::randn_like(x, rng) * sigmas[i + 1];
        }
    }
    return x;
}

static sd::Tensor<float> sample_ipndm(denoise_cb_t model,
                                      sd::Tensor<float> x,
                                      const std::vector<float>& sigmas) {
    const int max_order                 = 4;
    std::vector<sd::Tensor<float>> hist = {};

    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        float sigma      = sigmas[i];
        float sigma_next = sigmas[i + 1];

        auto denoised_opt = model(x, sigma, i + 1);
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised = std::move(denoised_opt);

        sd::Tensor<float> d_cur = (x - denoised) / sigma;
        int order               = std::min(max_order, i + 1);
        float dt                = sigma_next - sigma;

        switch (order) {
            case 1:
                x += d_cur * dt;
                break;
            case 2:
                x += ((3.f * d_cur - hist.back()) / 2.f) * dt;
                break;
            case 3:
                x += ((23.f * d_cur - 16.f * hist[hist.size() - 1] + 5.f * hist[hist.size() - 2]) / 12.f) * dt;
                break;
            case 4:
                x += ((55.f * d_cur - 59.f * hist[hist.size() - 1] + 37.f * hist[hist.size() - 2] - 9.f * hist[hist.size() - 3]) / 24.f) * dt;
                break;
        }

        if (hist.size() == static_cast<size_t>(max_order - 1)) {
            hist.erase(hist.begin());
        }
        hist.push_back(std::move(d_cur));
    }
    return x;
}

static sd::Tensor<float> sample_ipndm_v(denoise_cb_t model,
                                        sd::Tensor<float> x,
                                        const std::vector<float>& sigmas) {
    const int max_order                 = 4;
    std::vector<sd::Tensor<float>> hist = {};

    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        float sigma  = sigmas[i];
        float t_next = sigmas[i + 1];

        auto denoised_opt = model(x, sigma, i + 1);
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised = std::move(denoised_opt);

        sd::Tensor<float> d_cur = (x - denoised) / sigma;
        int order               = std::min(max_order, i + 1);
        float h_n               = t_next - sigma;
        float h_n_1             = (i > 0) ? (sigma - sigmas[i - 1]) : h_n;

        switch (order) {
            case 1:
                x += d_cur * h_n;
                break;
            case 2:
                x += (((2.f + (h_n / h_n_1)) * d_cur - (h_n / h_n_1) * hist.back()) / 2.f) * h_n;
                break;
            case 3:
                x += ((23.f * d_cur - 16.f * hist[hist.size() - 1] + 5.f * hist[hist.size() - 2]) / 12.f) * h_n;
                break;
            case 4:
                x += ((55.f * d_cur - 59.f * hist[hist.size() - 1] + 37.f * hist[hist.size() - 2] - 9.f * hist[hist.size() - 3]) / 24.f) * h_n;
                break;
        }

        if (hist.size() == static_cast<size_t>(max_order - 1)) {
            hist.erase(hist.begin());
        }
        hist.push_back(std::move(d_cur));
    }
    return x;
}

static sd::Tensor<float> sample_res_multistep(denoise_cb_t model,
                                              sd::Tensor<float> x,
                                              const std::vector<float>& sigmas,
                                              std::shared_ptr<RNG> rng,
                                              bool is_flow_denoiser,
                                              float eta) {
    sd::Tensor<float> old_denoised = x;
    bool have_old_sigma            = false;
    float old_sigma_down           = 0.0f;

    auto t_fn     = [](float sigma) -> float { return -logf(sigma); };
    auto sigma_fn = [](float t) -> float { return expf(-t); };
    auto phi1_fn  = [](float t) -> float {
        if (fabsf(t) < 1e-6f) {
            return 1.0f + t * 0.5f + (t * t) / 6.0f;
        }
        return (expf(t) - 1.0f) / t;
    };
    auto phi2_fn = [&](float t) -> float {
        if (fabsf(t) < 1e-6f) {
            return 0.5f + t / 6.0f + (t * t) / 24.0f;
        }
        float phi1_val = phi1_fn(t);
        return (phi1_val - 1.0f) / t;
    };

    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        auto denoised_opt = model(x, sigmas[i], i + 1);
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised = std::move(denoised_opt);

        float sigma_from            = sigmas[i];
        float sigma_to              = sigmas[i + 1];
        auto [sigma_down, sigma_up, alpha_scale] = get_ancestral_step(sigma_from, sigma_to, eta, is_flow_denoiser);

        if (sigma_down == 0.0f || !have_old_sigma) {
            x += ((x - denoised) / sigma_from) * (sigma_down - sigma_from);
        } else {
            float t      = t_fn(sigma_from);
            float t_old  = t_fn(old_sigma_down);
            float t_next = t_fn(sigma_down);
            float t_prev = t_fn(sigmas[i - 1]);
            float h      = t_next - t;
            float c2     = (t_prev - t_old) / h;

            float phi1_val = phi1_fn(-h);
            float phi2_val = phi2_fn(-h);
            float b1       = phi1_val - phi2_val / c2;
            float b2       = phi2_val / c2;

            if (!std::isfinite(b1)) {
                b1 = 0.0f;
            }
            if (!std::isfinite(b2)) {
                b2 = 0.0f;
            }

            x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised);
        }

        if (sigma_to > 0.0f && sigma_up > 0.0f) {
            if (is_flow_denoiser) {
                x *= alpha_scale;
            }
            x += sd::Tensor<float>::randn_like(x, rng) * sigma_up;
        }

        old_denoised   = denoised;
        old_sigma_down = sigma_down;
        have_old_sigma = true;
    }
    return x;
}

static sd::Tensor<float> sample_res_2s(denoise_cb_t model,
                                       sd::Tensor<float> x,
                                       const std::vector<float>& sigmas,
                                       std::shared_ptr<RNG> rng,
                                       bool is_flow_denoiser,
                                       float eta) {
    const float c2 = 0.5f;
    auto t_fn      = [](float sigma) -> float { return -logf(sigma); };
    auto phi1_fn   = [](float t) -> float {
        if (fabsf(t) < 1e-6f) {
            return 1.0f + t * 0.5f + (t * t) / 6.0f;
        }
        return (expf(t) - 1.0f) / t;
    };
    auto phi2_fn = [&](float t) -> float {
        if (fabsf(t) < 1e-6f) {
            return 0.5f + t / 6.0f + (t * t) / 24.0f;
        }
        float phi1_val = phi1_fn(t);
        return (phi1_val - 1.0f) / t;
    };

    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        float sigma_from = sigmas[i];
        float sigma_to   = sigmas[i + 1];

        auto denoised_opt = model(x, sigma_from, -(i + 1));
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised = std::move(denoised_opt);

        auto [sigma_down, sigma_up, alpha_scale] = get_ancestral_step(sigma_from, sigma_to, eta, is_flow_denoiser);

        sd::Tensor<float> x0 = x;
        if (sigma_down == 0.0f || sigma_from == 0.0f) {
            x = denoised;
        } else {
            float t      = t_fn(sigma_from);
            float t_next = t_fn(sigma_down);
            float h      = t_next - t;

            float a21      = c2 * phi1_fn(-h * c2);
            float phi1_val = phi1_fn(-h);
            float phi2_val = phi2_fn(-h);
            float b2       = phi2_val / c2;
            float b1       = phi1_val - b2;

            float sigma_c2         = expf(-(t + h * c2));
            sd::Tensor<float> eps1 = denoised - x0;
            sd::Tensor<float> x2   = x0 + eps1 * (h * a21);

            auto denoised2_opt = model(x2, sigma_c2, i + 1);
            if (denoised2_opt.empty()) {
                return {};
            }
            sd::Tensor<float> denoised2 = std::move(denoised2_opt);
            sd::Tensor<float> eps2      = denoised2 - x0;
            x                           = x0 + h * (b1 * eps1 + b2 * eps2);
        }

        if (sigma_to > 0.0f && sigma_up > 0.0f) {
            if (is_flow_denoiser) {
                x *= alpha_scale;
            }
            x += sd::Tensor<float>::randn_like(x, rng) * sigma_up;
        }
    }
    return x;
}

static float er_sde_noise_scaler(float value) {
    if (value <= 0.0f) {
        return 0.0f;
    }
    return value * (std::exp(std::pow(value, 0.3f)) + 10.0f);
}

static sd::Tensor<float> sample_er_sde(denoise_cb_t model,
                                       sd::Tensor<float> x,
                                       std::vector<float> sigmas,
                                       std::shared_ptr<RNG> rng,
                                       float s_noise,
                                       bool is_flow_denoiser,
                                       int max_stage = 3) {
    const float num_integration_points = 200.0f;
    max_stage                          = std::max(1, std::min(max_stage, 3));

    dpmpp_offset_first_sigma_for_snr(sigmas, is_flow_denoiser);
    std::vector<float> er_lambdas(sigmas.size(), 0.0f);
    for (size_t i = 0; i < sigmas.size(); ++i) {
        er_lambdas[i] = std::exp(-dpmpp_sigma_to_half_log_snr(sigmas[i], is_flow_denoiser));
    }

    sd::Tensor<float> old_denoised;
    sd::Tensor<float> old_denoised_d;

    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        const float sigma = sigmas[i];
        auto denoised_opt = model(x, sigma, i + 1);
        if (denoised_opt.empty()) {
            return {};
        }
        sd::Tensor<float> denoised = std::move(denoised_opt);
        const int stage_used       = std::min(max_stage, i + 1);

        if (sigmas[i + 1] == 0.0f) {
            x = denoised;
        } else {
            const float er_lambda_s = er_lambdas[i];
            const float er_lambda_t = er_lambdas[i + 1];
            if (er_lambda_s <= 0.0f || er_lambda_t <= 0.0f) {
                x = denoised;
                old_denoised = denoised;
                continue;
            }

            const float alpha_s = sigmas[i] / er_lambda_s;
            const float alpha_t = sigmas[i + 1] / er_lambda_t;
            const float r_alpha = alpha_t / alpha_s;
            const float scaled_s = er_sde_noise_scaler(er_lambda_s);
            const float scaled_t = er_sde_noise_scaler(er_lambda_t);
            const float r = scaled_s != 0.0f ? scaled_t / scaled_s : 0.0f;

            x = (r_alpha * r) * x + (alpha_t * (1.0f - r)) * denoised;

            if (stage_used >= 2 && !old_denoised.empty()) {
                const float dt               = er_lambda_t - er_lambda_s;
                const float lambda_step_size = -dt / num_integration_points;
                float sum_inv_scaled         = 0.0f;
                float sum_u_scaled           = 0.0f;
                for (int point = 0; point < static_cast<int>(num_integration_points); ++point) {
                    const float lambda_pos = er_lambda_t + static_cast<float>(point) * lambda_step_size;
                    const float scaled_pos = er_sde_noise_scaler(lambda_pos);
                    if (scaled_pos != 0.0f) {
                        sum_inv_scaled += 1.0f / scaled_pos;
                        sum_u_scaled += (lambda_pos - er_lambda_s) / scaled_pos;
                    }
                }
                const float s = sum_inv_scaled * lambda_step_size;
                sd::Tensor<float> denoised_d =
                    (denoised - old_denoised) * (1.0f / (er_lambda_s - er_lambdas[i - 1]));
                x += denoised_d * (alpha_t * (dt + s * scaled_t));

                if (stage_used >= 3 && !old_denoised_d.empty()) {
                    const float denom = (er_lambda_s - er_lambdas[i - 2]) * 0.5f;
                    if (denom != 0.0f) {
                        const float s_u = sum_u_scaled * lambda_step_size;
                        sd::Tensor<float> denoised_u = (denoised_d - old_denoised_d) * (1.0f / denom);
                        x += denoised_u * (alpha_t * ((dt * dt) * 0.5f + s_u * scaled_t));
                    }
                }
                old_denoised_d = denoised_d;
            }

            if (s_noise > 0.0f) {
                const float variance = er_lambda_t * er_lambda_t - er_lambda_s * er_lambda_s * r * r;
                const float noise_scale = alpha_t * s_noise * std::sqrt(std::max(variance, 0.0f));
                if (noise_scale > 0.0f) {
                    x += sd::Tensor<float>::randn_like(x, rng) * noise_scale;
                }
            }
        }
        old_denoised = denoised;
    }
    return x;
}

static sd::Tensor<float> sample_ddim_trailing(denoise_cb_t model,
                                              sd::Tensor<float> x,
                                              const std::vector<float>& sigmas,
                                              std::shared_ptr<RNG> rng,
                                              float eta) {
    float beta_start = 0.00085f;
    float beta_end   = 0.0120f;
    std::vector<double> alphas_cumprod(TIMESTEPS);
    std::vector<double> compvis_sigmas(TIMESTEPS);
    for (int i = 0; i < TIMESTEPS; i++) {
        alphas_cumprod[i] =
            (i == 0 ? 1.0f : alphas_cumprod[i - 1]) *
            (1.0f -
             std::pow(sqrtf(beta_start) +
                          (sqrtf(beta_end) - sqrtf(beta_start)) *
                              ((float)i / (TIMESTEPS - 1)),
                      2));
        compvis_sigmas[i] =
            std::sqrt((1 - alphas_cumprod[i]) / alphas_cumprod[i]);
    }

    int steps = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        int timestep      = static_cast<int>(roundf(TIMESTEPS - i * ((float)TIMESTEPS / steps))) - 1;
        int prev_timestep = timestep - TIMESTEPS / steps;
        float sigma       = static_cast<float>(compvis_sigmas[timestep]);
        if (i == 0) {
            x *= std::sqrt(sigma * sigma + 1) / sigma;
        } else {
            x *= std::sqrt(sigma * sigma + 1);
        }

        auto model_output_opt = model(x, sigma, i + 1);
        if (model_output_opt.empty()) {
            return {};
        }
        sd::Tensor<float> model_output = std::move(model_output_opt);
        model_output                   = (x - model_output) * (1.0f / sigma);

        float alpha_prod_t      = static_cast<float>(alphas_cumprod[timestep]);
        float alpha_prod_t_prev = static_cast<float>(prev_timestep >= 0 ? alphas_cumprod[prev_timestep] : alphas_cumprod[0]);
        float beta_prod_t       = 1.0f - alpha_prod_t;

        sd::Tensor<float> pred_original_sample = ((x / std::sqrt(sigma * sigma + 1)) -
                                                  std::sqrt(beta_prod_t) * model_output) *
                                                 (1.0f / std::sqrt(alpha_prod_t));

        float beta_prod_t_prev = 1.0f - alpha_prod_t_prev;
        float variance         = (beta_prod_t_prev / beta_prod_t) *
                         (1.0f - alpha_prod_t / alpha_prod_t_prev);
        float std_dev_t = eta * std::sqrt(variance);

        x = std::sqrt(alpha_prod_t_prev) * pred_original_sample +
            std::sqrt(1.0f - alpha_prod_t_prev - std::pow(std_dev_t, 2)) * model_output;

        if (eta > 0) {
            x += std_dev_t * sd::Tensor<float>::randn_like(x, rng);
        }
    }
    return x;
}

static sd::Tensor<float> sample_tcd(denoise_cb_t model,
                                    sd::Tensor<float> x,
                                    const std::vector<float>& sigmas,
                                    std::shared_ptr<RNG> rng,
                                    float eta) {
    float beta_start = 0.00085f;
    float beta_end   = 0.0120f;
    std::vector<double> alphas_cumprod(TIMESTEPS);
    std::vector<double> compvis_sigmas(TIMESTEPS);
    for (int i = 0; i < TIMESTEPS; i++) {
        alphas_cumprod[i] =
            (i == 0 ? 1.0f : alphas_cumprod[i - 1]) *
            (1.0f -
             std::pow(sqrtf(beta_start) +
                          (sqrtf(beta_end) - sqrtf(beta_start)) *
                              ((float)i / (TIMESTEPS - 1)),
                      2));
        compvis_sigmas[i] =
            std::sqrt((1 - alphas_cumprod[i]) / alphas_cumprod[i]);
    }

    int original_steps = 50;
    int steps          = static_cast<int>(sigmas.size()) - 1;
    for (int i = 0; i < steps; i++) {
        int timestep      = TIMESTEPS - 1 - (TIMESTEPS / original_steps) * (int)floor(i * ((float)original_steps / steps));
        int prev_timestep = i >= steps - 1 ? 0 : TIMESTEPS - 1 - (TIMESTEPS / original_steps) * (int)floor((i + 1) * ((float)original_steps / steps));
        int timestep_s    = (int)floor((1 - eta) * prev_timestep);
        float sigma       = static_cast<float>(compvis_sigmas[timestep]);

        if (i == 0) {
            x *= std::sqrt(sigma * sigma + 1) / sigma;
        } else {
            x *= std::sqrt(sigma * sigma + 1);
        }

        auto model_output_opt = model(x, sigma, i + 1);
        if (model_output_opt.empty()) {
            return {};
        }
        sd::Tensor<float> model_output = std::move(model_output_opt);
        model_output                   = (x - model_output) * (1.0f / sigma);

        float alpha_prod_t      = static_cast<float>(alphas_cumprod[timestep]);
        float beta_prod_t       = 1.0f - alpha_prod_t;
        float alpha_prod_t_prev = static_cast<float>(prev_timestep >= 0 ? alphas_cumprod[prev_timestep] : alphas_cumprod[0]);
        float alpha_prod_s      = static_cast<float>(alphas_cumprod[timestep_s]);
        float beta_prod_s       = 1.0f - alpha_prod_s;

        sd::Tensor<float> pred_original_sample = ((x / std::sqrt(sigma * sigma + 1)) -
                                                  std::sqrt(beta_prod_t) * model_output) *
                                                 (1.0f / std::sqrt(alpha_prod_t));

        x = std::sqrt(alpha_prod_s) * pred_original_sample +
            std::sqrt(beta_prod_s) * model_output;

        if (eta > 0 && i != steps - 1) {
            x = std::sqrt(alpha_prod_t_prev / alpha_prod_s) * x +
                std::sqrt(1.0f - alpha_prod_t_prev / alpha_prod_s) * sd::Tensor<float>::randn_like(x, rng);
        }
    }
    return x;
}

// k diffusion reverse ODE: dx = (x - D(x;\sigma)) / \sigma dt; \sigma(t) = t
static sd::Tensor<float> sample_k_diffusion(sample_method_t method,
                                            denoise_cb_t model,
                                            sd::Tensor<float> x,
                                            std::vector<float> sigmas,
                                            std::shared_ptr<RNG> rng,
                                            float eta,
                                            float s_noise,
                                            float dpmpp_sde_r,
                                            dpmpp_sde_solver_t dpmpp_sde_solver,
                                            bool is_flow_denoiser) {
    switch (method) {
        case EULER_A_SAMPLE_METHOD:
            if (is_flow_denoiser)
                return sample_euler_flow(model, std::move(x), sigmas, rng, eta);
            else
                return sample_euler_ancestral(model, std::move(x), sigmas, rng, eta);
        case EULER_SAMPLE_METHOD:
            return sample_euler(model, std::move(x), sigmas);
        case HEUN_SAMPLE_METHOD:
            return sample_heun(model, std::move(x), sigmas);
        case DPM2_SAMPLE_METHOD:
            return sample_dpm2(model, std::move(x), sigmas);
        case DPMPP2S_A_SAMPLE_METHOD:
            return sample_dpmpp_2s_ancestral(model, std::move(x), sigmas, rng, eta);
        case DPMPP2M_SAMPLE_METHOD:
            return sample_dpmpp_2m(model, std::move(x), sigmas);
        case DPMPP2Mv2_SAMPLE_METHOD:
            return sample_dpmpp_2m_v2(model, std::move(x), sigmas);
        case DPMPP_SDE_SAMPLE_METHOD:
        case DPMPP_SDE_GPU_SAMPLE_METHOD:
            return sample_dpmpp_sde(model, std::move(x), sigmas, rng, eta, s_noise, dpmpp_sde_r, is_flow_denoiser);
        case DPMPP2M_SDE_SAMPLE_METHOD:
        case DPMPP2M_SDE_GPU_SAMPLE_METHOD:
        case DPMPP2M_SDE_HEUN_SAMPLE_METHOD:
        case DPMPP2M_SDE_HEUN_GPU_SAMPLE_METHOD:
            return sample_dpmpp_2m_sde(model, std::move(x), sigmas, rng, eta, s_noise, dpmpp_sde_solver, is_flow_denoiser);
        case DPMPP3M_SDE_SAMPLE_METHOD:
        case DPMPP3M_SDE_GPU_SAMPLE_METHOD:
            return sample_dpmpp_3m_sde(model, std::move(x), sigmas, rng, eta, s_noise, is_flow_denoiser);
        case LCM_SAMPLE_METHOD:
            return sample_lcm(model, std::move(x), sigmas, rng);
        case IPNDM_SAMPLE_METHOD:
            return sample_ipndm(model, std::move(x), sigmas);
        case IPNDM_V_SAMPLE_METHOD:
            return sample_ipndm_v(model, std::move(x), sigmas);
        case RES_MULTISTEP_SAMPLE_METHOD:
            return sample_res_multistep(model, std::move(x), sigmas, rng, is_flow_denoiser, eta);
        case RES_2S_SAMPLE_METHOD:
            return sample_res_2s(model, std::move(x), sigmas, rng, is_flow_denoiser, eta);
        case ER_SDE_SAMPLE_METHOD:
            return sample_er_sde(model, std::move(x), sigmas, rng, s_noise, is_flow_denoiser);
        case DDIM_TRAILING_SAMPLE_METHOD:
            return sample_ddim_trailing(model, std::move(x), sigmas, rng, eta);
        case TCD_SAMPLE_METHOD:
            return sample_tcd(model, std::move(x), sigmas, rng, eta);
        default:
            return {};
    }
}

#endif  // __DENOISER_HPP__
