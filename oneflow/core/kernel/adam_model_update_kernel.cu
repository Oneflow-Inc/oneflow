#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/adam_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(int64_t n, int64_t batch_size, T learning_rate, T l1, T l2, T beta1,
                               T beta2, T epsilon, bool do_bias_correction, const T* beta1_t,
                               const T* beta2_t, const T* model_diff, T* model, T* m, T* v) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    m[i] = beta1 * m[i] + (1 - beta1) * model_diff[i];
    v[i] = beta2 * v[i] + (1 - beta2) * model_diff[i] * model_diff[i];
    if (do_bias_correction) {
      learning_rate = learning_rate * sqrt(1 - (*beta2_t)) / (1 - (*beta1_t));
    }
    T reg_diff = RegularizeDiff(m[i] / (sqrt(v[i]) + epsilon), batch_size, l1, l2, model[i]);
    model[i] = model[i] - learning_rate * reg_diff;
  }
}

}  // namespace

template<typename T>
class AdamMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, int64_t batch_size, T learning_rate, T l1,
                          T l2, T beta1, T beta2, T epsilon, bool do_bias_correction,
                          const T* beta1_t, const T* beta2_t, const T* model_diff, T* model, T* m,
                          T* v) {
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, batch_size, learning_rate, l1, l2, beta1, beta2, epsilon, do_bias_correction, beta1_t,
        beta2_t, model_diff, model, m, v);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class AdamMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
