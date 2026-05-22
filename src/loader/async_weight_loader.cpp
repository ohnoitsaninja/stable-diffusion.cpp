#include "async_weight_loader.h"

#include "loader_stats.h"

#include <chrono>

#include "ggml.h"

#if defined(SD_CUDA_THREADED_WEIGHT_LOADER) && defined(SD_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace sd {
namespace loader {

AsyncWeightLoader::AsyncWeightLoader() {
#if defined(SD_CUDA_THREADED_WEIGHT_LOADER) && defined(SD_USE_CUDA)
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err == cudaSuccess) {
        stream_ = stream;
    } else {
        (void)cudaGetLastError();
        note_fallback_reason(LoaderFallbackReason::unsupported_backend);
    }
#endif
}

AsyncWeightLoader::~AsyncWeightLoader() {
    synchronize();
#if defined(SD_CUDA_THREADED_WEIGHT_LOADER) && defined(SD_USE_CUDA)
    if (stream_ != nullptr) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
        stream_ = nullptr;
    }
#endif
}

bool AsyncWeightLoader::available() const {
    return stream_ != nullptr;
}

bool AsyncWeightLoader::upload(ggml_backend_t backend, ggml_tensor* tensor, const void* data, size_t offset, size_t bytes) {
    if (tensor == nullptr || data == nullptr || bytes == 0) {
        return false;
    }

#if defined(SD_CUDA_THREADED_WEIGHT_LOADER) && defined(SD_USE_CUDA)
    if (stream_ != nullptr && tensor->data != nullptr) {
        cudaPointerAttributes attributes{};
        cudaError_t attr_err = cudaPointerGetAttributes(&attributes, tensor->data);
        if (attr_err == cudaSuccess && attributes.type == cudaMemoryTypeDevice) {
            char* dst = static_cast<char*>(tensor->data) + offset;
            cudaEvent_t start_event = nullptr;
            cudaEvent_t stop_event  = nullptr;
            bool event_ok = cudaEventCreate(&start_event) == cudaSuccess &&
                            cudaEventCreate(&stop_event) == cudaSuccess &&
                            cudaEventRecord(start_event, static_cast<cudaStream_t>(stream_)) == cudaSuccess;
            cudaError_t copy_err = cudaMemcpyAsync(dst,
                                                   data,
                                                   bytes,
                                                   cudaMemcpyHostToDevice,
                                                   static_cast<cudaStream_t>(stream_));
            if (copy_err == cudaSuccess) {
                if (event_ok && cudaEventRecord(stop_event, static_cast<cudaStream_t>(stream_)) == cudaSuccess) {
                    pending_events_.push_back(start_event);
                    pending_events_.push_back(stop_event);
                } else {
                    if (start_event != nullptr) {
                        cudaEventDestroy(start_event);
                    }
                    if (stop_event != nullptr) {
                        cudaEventDestroy(stop_event);
                    }
                    (void)cudaGetLastError();
                }
                pending_bytes_ += static_cast<uint64_t>(bytes);
                return true;
            }
            if (start_event != nullptr) {
                cudaEventDestroy(start_event);
            }
            if (stop_event != nullptr) {
                cudaEventDestroy(stop_event);
            }
            (void)cudaGetLastError();
        } else {
            (void)cudaGetLastError();
        }
    }
#endif

    note_fallback(LoaderFallbackReason::unsupported_backend, static_cast<uint64_t>(bytes));
    if (backend != nullptr) {
        ggml_backend_tensor_set_async(backend, tensor, data, offset, bytes);
        ggml_backend_synchronize(backend);
    } else {
        ggml_backend_tensor_set(tensor, data, offset, bytes);
    }
    return false;
}

void AsyncWeightLoader::synchronize() {
#if defined(SD_CUDA_THREADED_WEIGHT_LOADER) && defined(SD_USE_CUDA)
    if (stream_ != nullptr && pending_bytes_ > 0) {
        const auto start = std::chrono::steady_clock::now();
        note_stream_synchronize();
        cudaError_t err = cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
        const auto end = std::chrono::steady_clock::now();
        if (err == cudaSuccess) {
            const double ms = std::chrono::duration<double, std::milli>(end - start).count();
            double event_ms = 0.0;
            for (size_t i = 0; i + 1 < pending_events_.size(); i += 2) {
                float elapsed = 0.0f;
                cudaEvent_t begin = static_cast<cudaEvent_t>(pending_events_[i]);
                cudaEvent_t finish = static_cast<cudaEvent_t>(pending_events_[i + 1]);
                if (cudaEventElapsedTime(&elapsed, begin, finish) == cudaSuccess) {
                    event_ms += elapsed;
                } else {
                    (void)cudaGetLastError();
                }
                cudaEventDestroy(begin);
                cudaEventDestroy(finish);
            }
            pending_events_.clear();
            add_h2d(pending_bytes_, ms, event_ms);
        } else {
            (void)cudaGetLastError();
            note_fallback(LoaderFallbackReason::other, pending_bytes_);
        }
        pending_bytes_ = 0;
    }
#endif
}

}  // namespace loader
}  // namespace sd
