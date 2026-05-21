#ifndef SD_ASYNC_WEIGHT_LOADER_H
#define SD_ASYNC_WEIGHT_LOADER_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ggml-backend.h"

namespace sd {
namespace loader {

class AsyncWeightLoader {
public:
    AsyncWeightLoader();
    ~AsyncWeightLoader();

    AsyncWeightLoader(const AsyncWeightLoader&) = delete;
    AsyncWeightLoader& operator=(const AsyncWeightLoader&) = delete;

    bool available() const;
    bool upload(ggml_backend_t backend, ggml_tensor* tensor, const void* data, size_t offset, size_t bytes);
    void synchronize();

private:
    void* stream_ = nullptr;
    uint64_t pending_bytes_ = 0;
    std::vector<void*> pending_events_;
};

}  // namespace loader
}  // namespace sd

#endif  // SD_ASYNC_WEIGHT_LOADER_H
