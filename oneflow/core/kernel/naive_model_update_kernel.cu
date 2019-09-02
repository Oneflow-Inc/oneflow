#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/naive_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(const int64_t n, const T* batch_instance_num_ptr,
                               const float* learning_rate, T l1, T l2, const T* model_diff,
                               T* model) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T reg_diff = RegularizeDiff(model_diff[i], *batch_instance_num_ptr, l1, l2, model[i]);
    model[i] = model[i] - *learning_rate * reg_diff;
  }
}

}  // namespace

template<typename T>
class NaiveMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, const int64_t n, const T* batch_instance_num_ptr,
                          const float* learning_rate, T l1, T l2, const T* model_diff, T* model) {
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, batch_instance_num_ptr, learning_rate, l1, l2, model_diff, model);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class NaiveMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
