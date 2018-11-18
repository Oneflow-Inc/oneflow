#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/adam_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateMomentEstimateGpu(int64_t n, T beta, int32_t p, const T* model_diff,
                                        const T* beta_t, T* momentum) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    // Update biased moment estimate
    momentum[i] = beta * momentum[i] + (1 - beta) * std::pow(model_diff[i], p);
    // Correct deviation of moment estimate
    momentum[i] = momentum[i] / (1 - *beta_t);
  }
}

template<typename T>
__global__ void UpdateModelGpu(int64_t n, const T* batch_instance_num_ptr, T learning_rate, T l1,
                               T l2, T epsilon, T* model_diff, T* model, T* m, T* v) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    model_diff[i] = m[i] / (std::sqrt(v[i]) + epsilon);
    T reg_diff = RegularizeDiff(model_diff[i], *batch_instance_num_ptr, l1, l2, model[i]);
    model[i] = model[i] - learning_rate * reg_diff;
  }
}

}  // namespace

template<typename T>
class AdamMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const T* batch_instance_num_ptr,
                          T learning_rate, T l1, T l2, T beta1, T beta2, T epsilon,
                          int64_t next_model_vid, const T* beta1_t, const T* beta2_t, T* model_diff,
                          T* model, T* m, T* v) {
    // first-order moment
    UpdateMomentEstimateGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, beta1, 1, model_diff, beta1_t, m);
    // second-order moment
    UpdateMomentEstimateGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, beta2, 2, model_diff, beta2_t, v);
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, batch_instance_num_ptr, learning_rate, l1, l2, epsilon, model_diff, model, m, v);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class AdamMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
