#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(int64_t n, T learning_rate, T decay_rate, T epsilon, T l1, T l2,
                               const T* model_diff, T* model, T* mean_square) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T reg_diff = RegularizeDiff(model_diff[i], l1, l2, model[i]);
    mean_square[i] = (1 - decay_rate) * reg_diff * reg_diff + decay_rate * mean_square[i];
    model[i] = model[i] - learning_rate * reg_diff / std::sqrt(mean_square[i] + epsilon);
  }
}

}  // namespace

template<typename T>
class RMSPropMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, T learning_rate, T decay_rate, T epsilon, T l1,
                          T l2, const T* model_diff, T* model, T* mean_square) {
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, learning_rate, decay_rate, epsilon, l1, l2, model_diff, model, mean_square);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class RMSPropMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
