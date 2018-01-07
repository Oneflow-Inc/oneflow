#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/momentum_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(const int64_t n, const T beta, const T alpha,
                               const T* model_diff, const T* pre_model,
                               T* momentum, T* model) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    momentum[i] = beta * momentum[i] + alpha * model_diff[i];
    model[i] = pre_model[i] + momentum[i];
  }
}

}  // namespace

template<typename T>
class MomentumMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, const int64_t n, const T beta,
                          const T alpha, const T* model_diff,
                          const T* pre_model, T* momentum, T* model) {
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                        ctx->cuda_stream()>>>(n, beta, alpha, model_diff,
                                              pre_model, momentum, model);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class MomentumMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
