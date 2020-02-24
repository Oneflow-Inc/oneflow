#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/lazy_adam_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(int64_t n, const float* learning_rate, T weight_decay, T beta1,
                               T beta2, T epsilon, const T* beta1_t, const T* beta2_t,
                               T* model_diff, T* model, T* m, T* v) {
  const float local_learning_rate = *learning_rate * sqrt(1 - (*beta2_t)) / (1 - (*beta1_t));
  CUDA_1D_KERNEL_LOOP(i, n) {
    T model_diff_val = model_diff[i];
    if (abs(model_diff_val) < 1e-12) { continue; }
    m[i] = beta1 * m[i] + (1 - beta1) * model_diff_val;
    v[i] = beta2 * v[i] + (1 - beta2) * model_diff_val * model_diff_val;
    model[i] = model[i] - local_learning_rate * m[i] / (sqrt(v[i]) + epsilon);
  }
}

template<typename T>
__global__ void UpdateBetaTGpu(const int64_t* train_step, const T beta1, const T beta2, T* beta1_t,
                               T* beta2_t) {
  *beta1_t *= beta1;
  *beta2_t *= beta2;
}

}  // namespace

template<typename T>
class LazyAdamMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const float* learning_rate, T weight_decay,
                          T beta1, T beta2, T epsilon, const int64_t* train_step, T* beta1_t,
                          T* beta2_t, T* model_diff, T* model, T* m, T* v) {
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, learning_rate, weight_decay, beta1, beta2, epsilon, beta1_t, beta2_t, model_diff, model,
        m, v);
    UpdateBetaTGpu<T><<<1, 1, 0, ctx->cuda_stream()>>>(train_step, beta1, beta2, beta1_t, beta2_t);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class LazyAdamMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
