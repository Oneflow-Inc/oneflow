#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void RegularizationGpu(const int64_t n, float l1, float l2,
                                  const T* x, T* y) {
  T zero = static_cast<T>(0);
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] += l1 * static_cast<T>((x[i] > zero) - (zero < x[i])) + x[i] * l2;
  }
}

}  // namespace

template<typename T>
class MdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void Regularization(DeviceCtx* ctx, int64_t n, float l1, float l2,
                             const T* model, T* model_diff_acc) {
    RegularizationGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(n, l1, l2, model, model_diff_acc);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class MdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
