#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(int64_t n, const int64_t* train_step, const float* learning_rate,
                               T decay_rate, T epsilon, T weight_decay, const T* model_diff,
                               T* model, T* mean_square) {
  const T cur_decay_rate = *train_step == 0 ? 0 : decay_rate;
  CUDA_1D_KERNEL_LOOP(i, n) {
    T model_diff_val = model_diff[i];
    mean_square[i] =
        (1 - cur_decay_rate) * model_diff_val * model_diff_val + cur_decay_rate * mean_square[i];
    model[i] = model[i] - *learning_rate * model_diff_val / std::sqrt(mean_square[i] + epsilon);
  }
}

}  // namespace

template<typename T>
class RMSPropMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const int64_t* train_step,
                          const float* learning_rate, T decay_rate, T epsilon, T weight_decay,
                          const T* model_diff, T* model, T* mean_square) {
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, train_step, learning_rate, decay_rate, epsilon, weight_decay, model_diff, model,
        mean_square);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class RMSPropMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
