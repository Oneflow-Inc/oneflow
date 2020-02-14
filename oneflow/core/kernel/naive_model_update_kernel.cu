#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/naive_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(const int64_t n, const float* learning_rate, T weight_decay,
                               const T* model_diff, T* model) {
  const float lr = *learning_rate;
  CUDA_1D_KERNEL_LOOP(i, n) {
    model[i] = model[i] - lr * (model_diff[i] + weight_decay * model[i]);
  }
}

}  // namespace

template<typename T>
class NaiveMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, const int64_t n, const float* learning_rate,
                          T weight_decay, const T* model_diff, T* model) {
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, learning_rate, weight_decay, model_diff, model);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class NaiveMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
