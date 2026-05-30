#include "lens_gptoss_tokenizer.hpp"

#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

struct U32Hash {
    size_t operator()(const std::u32string& s) const noexcept {
        size_t h = 1469598103934665603ull;
        for (char32_t c : s) {
            h ^= static_cast<size_t>(c);
            h *= 1099511628211ull;
        }
        return h;
    }
};

struct PairHash {
    size_t operator()(const std::pair<std::u32string, std::u32string>& p) const noexcept {
        U32Hash h;
        return h(p.first) ^ (h(p.second) + 0x9e3779b97f4a7c15ull + (h(p.first) << 6) + (h(p.first) >> 2));
    }
};

static bool is_ascii_letter(char32_t c) {
    return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z');
}

static bool is_ascii_number(char32_t c) {
    return c >= U'0' && c <= U'9';
}

static bool is_ascii_space(char32_t c) {
    return c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || c == U'\v' || c == U'\f';
}

static std::string cp_to_utf8(char32_t cp) {
    std::string out;
    if (cp <= 0x7f) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
    return out;
}

static std::vector<char32_t> utf8_to_cps(const std::string& s) {
    std::vector<char32_t> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        char32_t cp = 0;
        size_t extra = 0;
        if ((c & 0x80) == 0) {
            cp = c;
        } else if ((c & 0xe0) == 0xc0) {
            cp = c & 0x1f;
            extra = 1;
        } else if ((c & 0xf0) == 0xe0) {
            cp = c & 0x0f;
            extra = 2;
        } else if ((c & 0xf8) == 0xf0) {
            cp = c & 0x07;
            extra = 3;
        } else {
            ++i;
            continue;
        }
        if (i + extra >= s.size()) {
            break;
        }
        for (size_t j = 1; j <= extra; ++j) {
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + j]) & 0x3f);
        }
        out.push_back(cp);
        i += 1 + extra;
    }
    return out;
}

static std::u32string utf8_to_u32(const std::string& s) {
    const std::vector<char32_t> cps = utf8_to_cps(s);
    return std::u32string(cps.begin(), cps.end());
}

static std::unordered_map<unsigned char, std::u32string> make_byte_encoder() {
    std::unordered_map<unsigned char, std::u32string> byte_encoder;
    std::unordered_set<int> byte_set;
    auto add = [&](int b, int unicode) {
        byte_set.insert(b);
        byte_encoder[static_cast<unsigned char>(b)] = std::u32string(1, static_cast<char32_t>(unicode));
    };
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        add(b, b);
    }
    for (int b = 161; b <= 172; ++b) {
        add(b, b);
    }
    for (int b = 174; b <= 255; ++b) {
        add(b, b);
    }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (byte_set.find(b) == byte_set.end()) {
            add(b, n + 256);
            ++n;
        }
    }
    return byte_encoder;
}

static std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static bool try_contraction(const std::vector<char32_t>& cps, size_t i, std::string& suffix, size_t& consumed) {
    if (i >= cps.size() || cps[i] != U'\'') {
        return false;
    }
    std::string candidate;
    for (size_t j = i; j < cps.size() && j < i + 3; ++j) {
        candidate += cp_to_utf8(cps[j]);
        const std::string lower = lower_ascii(candidate);
        if (lower == "'s" || lower == "'t" || lower == "'m" || lower == "'d") {
            suffix = lower;
            consumed = j - i + 1;
            return true;
        }
        if (lower == "'re" || lower == "'ve" || lower == "'ll") {
            suffix = lower;
            consumed = j - i + 1;
            return true;
        }
    }
    return false;
}

// ASCII implementation of the GPT-OSS tokenizer regex. This covers the Lens
// no-tools prompt template and common English prompts used by the staged lane.
static std::vector<std::string> lens_gptoss_split(const std::string& text) {
    std::vector<std::string> tokens;
    const std::vector<char32_t> cps = utf8_to_cps(text);
    size_t i = 0;
    while (i < cps.size()) {
        const char32_t cp = cps[i];

        size_t start = i;
        bool has_prefix = false;
        if (cp != U'\r' && cp != U'\n' && !is_ascii_letter(cp) && !is_ascii_number(cp) &&
            i + 1 < cps.size() && is_ascii_letter(cps[i + 1])) {
            has_prefix = true;
            ++i;
        }
        if (i < cps.size() && is_ascii_letter(cps[i])) {
            ++i;
            while (i < cps.size() && is_ascii_letter(cps[i])) {
                ++i;
            }
            std::string suffix;
            size_t consumed = 0;
            if (try_contraction(cps, i, suffix, consumed)) {
                i += consumed;
            }
            std::string token;
            for (size_t j = start; j < i; ++j) {
                token += cp_to_utf8(cps[j]);
            }
            tokens.push_back(token);
            continue;
        }
        if (has_prefix) {
            i = start;
        }

        if (is_ascii_number(cp)) {
            std::string token;
            size_t count = 0;
            while (i < cps.size() && is_ascii_number(cps[i]) && count < 3) {
                token += cp_to_utf8(cps[i]);
                ++i;
                ++count;
            }
            tokens.push_back(token);
            continue;
        }

        if (cp == U' ' && i + 1 < cps.size() &&
            !is_ascii_space(cps[i + 1]) &&
            !is_ascii_letter(cps[i + 1]) &&
            !is_ascii_number(cps[i + 1])) {
            std::string token = " ";
            ++i;
            while (i < cps.size() && !is_ascii_space(cps[i]) && !is_ascii_letter(cps[i]) && !is_ascii_number(cps[i])) {
                token += cp_to_utf8(cps[i]);
                ++i;
            }
            while (i < cps.size() && (cps[i] == U'\r' || cps[i] == U'\n' || cps[i] == U'/')) {
                token += cp_to_utf8(cps[i]);
                ++i;
            }
            tokens.push_back(token);
            continue;
        }

        if (!is_ascii_space(cp) && !is_ascii_letter(cp) && !is_ascii_number(cp)) {
            std::string token;
            while (i < cps.size() && !is_ascii_space(cps[i]) && !is_ascii_letter(cps[i]) && !is_ascii_number(cps[i])) {
                token += cp_to_utf8(cps[i]);
                ++i;
            }
            while (i < cps.size() && (cps[i] == U'\r' || cps[i] == U'\n' || cps[i] == U'/')) {
                token += cp_to_utf8(cps[i]);
                ++i;
            }
            tokens.push_back(token);
            continue;
        }

        if (is_ascii_space(cp)) {
            std::string token;
            bool saw_newline = false;
            while (i < cps.size() && is_ascii_space(cps[i])) {
                token += cp_to_utf8(cps[i]);
                if (cps[i] == U'\r' || cps[i] == U'\n') {
                    saw_newline = true;
                } else if (saw_newline) {
                    break;
                }
                ++i;
            }
            tokens.push_back(token);
            continue;
        }

        tokens.push_back(cp_to_utf8(cp));
        ++i;
    }
    return tokens;
}

static std::vector<std::string> split_special(const std::string& text, const std::vector<std::string>& specials) {
    std::vector<std::string> parts;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t next = text.size();
        std::string matched;
        for (const std::string& token : specials) {
            const size_t found = text.find(token, pos);
            if (found != std::string::npos && found < next) {
                next = found;
                matched = token;
            }
        }
        if (next > pos) {
            parts.push_back(text.substr(pos, next - pos));
        }
        if (!matched.empty()) {
            parts.push_back(matched);
            pos = next + matched.size();
        } else {
            break;
        }
    }
    return parts;
}

static std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static std::string render_lens_prompt(const std::string& prompt, const std::string& current_date) {
    static const char* kDeveloperInstruction =
        "Describe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background.";
    static const char* kAssistantThinking = "Need to generate one image according to the description.";
    std::ostringstream out;
    out << "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
        << "Knowledge cutoff: 2024-06\n"
        << "Current date: " << current_date << "\n\n"
        << "Reasoning: medium\n\n"
        << "# Valid channels: analysis, commentary, final. Channel must be included for every message."
        << "<|end|><|start|>developer<|message|># Instructions\n\n"
        << kDeveloperInstruction << "\n\n"
        << "<|end|><|start|>user<|message|>" << prompt
        << "<|end|><|start|>assistant<|channel|>analysis<|message|>" << kAssistantThinking
        << "<|end|><|start|>assistant<|channel|>final<|message|>";
    return out.str();
}

}  // namespace

struct sd_lens_gptoss_tokenizer::Impl {
    std::unordered_map<std::u32string, int64_t, U32Hash> vocab;
    std::unordered_map<std::string, int64_t> special_ids;
    std::unordered_map<std::pair<std::u32string, std::u32string>, int, PairHash> ranks;
    std::unordered_map<unsigned char, std::u32string> byte_encoder;
    std::vector<std::string> special_tokens;
    int64_t pad_token_id = 199999;

    std::vector<int64_t> encode_text(const std::string& text) const {
        std::vector<int64_t> ids;
        for (const std::string& part : split_special(text, special_tokens)) {
            const auto special_it = special_ids.find(part);
            if (special_it != special_ids.end()) {
                ids.push_back(special_it->second);
                continue;
            }
            for (const std::string& token : lens_gptoss_split(part)) {
                std::u32string byte_token;
                for (unsigned char b : token) {
                    const auto it = byte_encoder.find(b);
                    if (it == byte_encoder.end()) {
                        throw std::runtime_error("byte encoder missing byte");
                    }
                    byte_token += it->second;
                }
                for (const std::u32string& piece : bpe(byte_token)) {
                    const auto id_it = vocab.find(piece);
                    if (id_it == vocab.end()) {
                        throw std::runtime_error("tokenizer vocab missing BPE piece");
                    }
                    ids.push_back(id_it->second);
                }
            }
        }
        return ids;
    }

    std::vector<std::u32string> bpe(const std::u32string& token) const {
        if (token.empty()) {
            return {};
        }
        std::vector<std::u32string> word;
        word.reserve(token.size());
        for (char32_t c : token) {
            word.emplace_back(1, c);
        }
        while (word.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best_index = word.size();
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                auto it = ranks.find({word[i], word[i + 1]});
                if (it != ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_index = i;
                }
            }
            if (best_index == word.size()) {
                break;
            }
            word[best_index] += word[best_index + 1];
            word.erase(word.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
        }
        return word;
    }
};

sd_lens_gptoss_tokenizer::sd_lens_gptoss_tokenizer() : impl(new Impl()) {}

sd_lens_gptoss_tokenizer::~sd_lens_gptoss_tokenizer() {
    delete impl;
}

bool sd_lens_gptoss_tokenizer::load(const std::string& tokenizer_dir, std::string* error) {
    try {
        Impl next;
        next.byte_encoder = make_byte_encoder();
        const std::filesystem::path tokenizer_json = std::filesystem::path(tokenizer_dir) / "tokenizer.json";
        const nlohmann::json root = nlohmann::json::parse(read_text_file(tokenizer_json));
        const nlohmann::json& vocab_json = root.at("model").at("vocab");
        next.vocab.reserve(vocab_json.size());
        for (auto it = vocab_json.begin(); it != vocab_json.end(); ++it) {
            next.vocab.emplace(utf8_to_u32(it.key()), it.value().get<int64_t>());
        }
        const nlohmann::json& merges = root.at("model").at("merges");
        next.ranks.reserve(merges.size());
        for (size_t i = 0; i < merges.size(); ++i) {
            const std::string a = merges[i][0].get<std::string>();
            const std::string b = merges[i][1].get<std::string>();
            next.ranks.emplace(std::make_pair(utf8_to_u32(a), utf8_to_u32(b)), static_cast<int>(i));
        }
        const nlohmann::json& added = root.at("added_tokens");
        for (const auto& item : added) {
            if (item.value("special", false)) {
                const std::string content = item.at("content").get<std::string>();
                const int64_t id = item.at("id").get<int64_t>();
                next.special_ids[content] = id;
                next.special_tokens.push_back(content);
                if (content == "<|endoftext|>") {
                    next.pad_token_id = id;
                }
            }
        }
        std::sort(next.special_tokens.begin(), next.special_tokens.end(), [](const std::string& a, const std::string& b) {
            return a.size() > b.size();
        });
        *impl = std::move(next);
        return true;
    } catch (const std::exception& e) {
        if (error != nullptr) {
            *error = e.what();
        }
        return false;
    }
}

bool sd_lens_gptoss_tokenizer::encode_prompt(const std::string& prompt,
                                             int max_sequence_length,
                                             const std::string& current_date,
                                             sd_lens_tokenized_prompt* out,
                                             std::string* error) const {
    try {
        if (out == nullptr) {
            throw std::runtime_error("encode_prompt requires an output pointer");
        }
        if (max_sequence_length <= 0) {
            throw std::runtime_error("max_sequence_length must be positive");
        }
        sd_lens_tokenized_prompt result;
        result.prompt = prompt;
        result.max_sequence_length = max_sequence_length;
        result.txt_offset = 97;
        result.pad_token_id = static_cast<int>(impl->pad_token_id);
        result.rendered_prompt = render_lens_prompt(prompt, current_date.empty() ? sd_lens_current_date_yyyy_mm_dd() : current_date);
        result.input_ids = impl->encode_text(result.rendered_prompt);
        if (static_cast<int>(result.input_ids.size()) > max_sequence_length) {
            result.input_ids.resize(static_cast<size_t>(max_sequence_length));
        }
        result.attention_mask.assign(result.input_ids.size(), 1);
        while (static_cast<int>(result.input_ids.size()) < max_sequence_length) {
            result.input_ids.push_back(impl->pad_token_id);
            result.attention_mask.push_back(0);
        }
        result.raw_seq_len = static_cast<int>(result.input_ids.size());
        result.trimmed_seq_len = std::max(0, result.raw_seq_len - result.txt_offset);
        *out = std::move(result);
        return true;
    } catch (const std::exception& e) {
        if (error != nullptr) {
            *error = e.what();
        }
        return false;
    }
}

std::string sd_lens_current_date_yyyy_mm_dd() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%d");
    return out.str();
}
