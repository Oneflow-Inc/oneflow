#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/lars_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(const int64_t n, const T lars_coefficient,
                               const T learning_rate, const T m,
                               const T weight_decay, T* model, T* momentum,
                               const T* model_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    // TODO
  }
}

}  // namespace

template<typename T>
class LARSMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(const KernelCtx& ctx, const int64_t n,
                          const T lars_coefficient, const T learning_rate,
                          const T m, const T weight_decay, T* model,
                          T* momentum, const T* model_diff) {
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                        ctx.device_ctx->cuda_stream()>>>(
        n, lars_coefficient, learning_rate, m, weight_decay, model, momentum,
        model_diff);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class LARSMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
