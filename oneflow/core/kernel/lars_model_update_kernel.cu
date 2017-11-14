#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/lars_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GetSquareGpu(const int64_t n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] * x[i]; }
}

template<typename T>
__global__ void UpdateModelGpu(const int64_t n, const T local_lr, const T m,
                               const T weight_decay, T* model, T* momentum,
                               const T* model_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    momentum[i] =
        m * momentum[i] + local_lr * (model_diff[i] + weight_decay * model[i]);
    model[i] = model[i] - momentum[i];
  }
}

template<typename T>
T GetNorm(DeviceCtx* ctx, const int64_t n, const T* data, T* temp) {
  GetSquareGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                    ctx->cuda_stream()>>>(n, data, temp);
  KernelUtil<DeviceType::kGPU, T>::Sum(ctx, n, temp, temp + n, temp + n + 1,
                                       (n - 1) * sizeof(T));
  return std::sqrt(*(temp + n) / n);
}

}  // namespace

template<typename T>
class LARSMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(const KernelCtx& ctx, const int64_t n,
                          const T lars_coefficient, const T learning_rate,
                          const T m, const T weight_decay, T* model,
                          T* momentum, T* temp, const T* model_diff) {
    T model_norm = GetNorm(ctx.device_ctx, n, model, temp);
    T model_diff_norm = GetNorm(ctx.device_ctx, n, model_diff, temp);

    const T local_lr = learning_rate * lars_coefficient * model_norm
                       / (model_diff_norm + weight_decay * model_norm);
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                        ctx.device_ctx->cuda_stream()>>>(
        n, local_lr, m, weight_decay, model, momentum, model_diff);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class LARSMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
