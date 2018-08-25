#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(int64_t n, int64_t batch_size, T learning_rate, T l1, T l2,
                               T momentum_beta, const T* pre_model, const T* model_diff,
                               T* momentum, T* model) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T reg_diff = RegularizeDiff(model_diff[i], batch_size, l1, l2, pre_model[i]);
    if (momentum_beta) {
      momentum[i] = momentum_beta * momentum[i] - learning_rate * reg_diff;
      model[i] = pre_model[i] + momentum[i];
    } else {
      model[i] = pre_model[i] - learning_rate * reg_diff;
    }
  }
}

}  // namespace

template<typename T>
void NormalMdUpdateKernelUtil<DeviceType::kGPU, T>::UpdateModel(
    DeviceCtx* ctx, int64_t n, int64_t batch_size, T learning_rate, T l1, T l2, T momentum_beta,
    const T* pre_model, const T* model_diff, T* momentum, T* model) {
  UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, batch_size, learning_rate, l1, l2, momentum_beta, pre_model, model_diff, momentum, model);
}

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class NormalMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
