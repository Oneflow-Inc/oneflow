#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(int64_t n, int64_t batch_size, T alpha, T learning_rate,
                               T decay_rate, T epsilon, T l1, T l2, const T* pre_model, T* model,
                               T* mean_square, const T* model_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T avg_model_diff = model_diff[i] / batch_size;
    mean_square[i] = alpha * avg_model_diff * avg_model_diff + decay_rate * mean_square[i];
    model[i] = pre_model[i] - learning_rate * avg_model_diff / std::sqrt(mean_square[i] + epsilon);
    model[i] -= l2 * learning_rate * pre_model[i];
    model[i] -= l1 * learning_rate * ((pre_model[i] >= 0) - (pre_model[i] <= 0));
  }
}

}  // namespace

template<typename T>
class RMSPropMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, int64_t batch_size, T alpha, T learning_rate,
                          T decay_rate, T epsilon, T l1, T l2, const T* pre_model, T* model,
                          T* mean_square, const T* model_diff) {
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, batch_size, alpha, learning_rate, decay_rate, epsilon, l1, l2, pre_model, model,
        mean_square, model_diff);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class RMSPropMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
