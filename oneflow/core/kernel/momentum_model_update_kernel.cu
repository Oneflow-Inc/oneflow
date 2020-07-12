#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/momentum_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(int64_t n, T beta, const int64_t* train_step,
                               const float* learning_rate, T weight_decay, const T* model_diff,
                               T* model, T* momentum) {
  const T lr = *learning_rate;
  CUDA_1D_KERNEL_LOOP(i, n) {
    T next_momentum = beta * momentum[i] - lr * model_diff[i];
    momentum[i] = next_momentum;
    model[i] = model[i] + next_momentum - lr * weight_decay * model[i];
  }
}

}  // namespace

template<typename T>
class MomentumMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, T beta, const int64_t* train_step,
                          const float* learning_rate, const T weight_decay, const T* model_diff,
                          T* model, T* momentum) {
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, beta, train_step, learning_rate, weight_decay, model_diff, model, momentum);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class MomentumMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
