#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/momentum_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(int64_t n, int64_t batch_size, T beta, T learning_rate, T l1, T l2,
                               const T* model_diff, const T* pre_model, T* momentum, T* model) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T avg_model_diff = model_diff[i] / batch_size;
    momentum[i] = beta * momentum[i] - learning_rate * (avg_model_diff + l2 * pre_model[i]);
    model[i] = pre_model[i] + momentum[i];
    model[i] -= l1 * ((pre_model[i] >= 0) - (pre_model[i] <= 0));
  }
}

}  // namespace

template<typename T>
class MomentumMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, int64_t batch_size, T beta, T learning_rate,
                          const T l1, const T l2, const T* model_diff, const T* pre_model,
                          T* momentum, T* model) {
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, batch_size, beta, learning_rate, l1, l2, model_diff, pre_model, momentum, model);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class MomentumMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
