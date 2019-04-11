#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/lamb_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateMomentEstimateGpu(int64_t n, const T* batch_instance_num_ptr, T l1, T l2,
                                        float beta1, float beta2, float epsilon,
                                        const float* beta1_t, const float* beta2_t, T* model_diff,
                                        T* model, T* m, T* v, T* fw_buf) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    model_diff[i] = model_diff[i] / *batch_instance_num_ptr;
    m[i] = beta1 * m[i] + (1 - beta1) * model_diff[i];
    v[i] = beta2 * v[i] + (1 - beta2) * (model_diff[i] * model_diff[i]);
    model_diff[i] = (m[i] / (1 - *beta1_t)) / sqrt(v[i] / (1 - *beta2_t) + epsilon)
                    + l1 * ((model[i] >= 0) - (model[i] <= 0)) + l2 * model[i];
  }
}

template<typename T>
__global__ void GetLocalLearningRateGpu(T learning_rate, T* fw_buf) {
  fw_buf[0] = sqrt(fw_buf[0]);
  fw_buf[1] = sqrt(fw_buf[1]);
  learning_rate = fw_buf[0] / fw_buf[1] * learning_rate;
}

template<typename T>
__global__ void UpdateModelGpu(int64_t n, T learning_rate, const T* model_diff, T* model) {
  CUDA_1D_KERNEL_LOOP(i, n) { model[i] = model[i] - learning_rate * model_diff[i]; }
}

}  // namespace

template<typename T>
class LAMBMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const T* batch_instance_num_ptr,
                          T learning_rate, T l1, T l2, float beta1, float beta2, float epsilon,
                          int64_t next_model_vid, const float* beta1_t, const float* beta2_t,
                          T* model_diff, T* model, T* m, T* v, T* fw_buf) {
    UpdateMomentEstimateGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, batch_instance_num_ptr, l1, l2, beta1, beta2, epsilon, beta1_t, beta2_t, model_diff,
            model, m, v, fw_buf);
    KernelUtil<DeviceType::kGPU, T>::Dot(ctx, n, model, 1, model, 1, &fw_buf[0]);
    KernelUtil<DeviceType::kGPU, T>::Dot(ctx, n, model_diff, 1, model_diff, 1, &fw_buf[1]);
    GetLocalLearningRateGpu<T><<<1, 1, 0, ctx->cuda_stream()>>>(learning_rate, fw_buf);
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, learning_rate, model_diff, model);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class LAMBMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
