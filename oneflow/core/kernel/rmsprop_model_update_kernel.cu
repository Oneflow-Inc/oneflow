#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(const int64_t n, const T alpha,
                               const T learning_rate, const T decay_rate,
                               const T epsilon, T* model, T* mean_square,
                               const T* model_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    mean_square[i] =
        alpha * model_diff[i] * model_diff[i] + decay_rate * mean_square[i];
    model[i] -=
        learning_rate * model_diff[i] / std::sqrt(mean_square[i] + epsilon);
  }
}

}  // namespace

template<typename T>
class RMSPropMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(const KernelCtx& ctx, const int64_t n, const T alpha,
                          const T learning_rate, const T decay_rate,
                          const T epsilon, T* model, T* mean_square,
                          const T* model_diff) {
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                        ctx.device_ctx->cuda_stream()>>>(
        n, alpha, learning_rate, decay_rate, epsilon, model, mean_square,
        model_diff);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class RMSPropMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
