#include "oneflow/core/kernel/maximum_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
namespace oneflow {

namespace {
template<typename T>
__global__ void CWiseMaxWithMaskGpu(const int64_t n, T* x, const T* y, const int y_idx,
                                    int32_t* mask) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (y[i] > x[i]) {
      x[i] = y[i];
      mask[i] = y_idx;
    }
  }
}

template<typename T>
__global__ void CWiseSetWithMaskGpu(const int64_t n, T* x, const T* y, const int x_idx,
                                    const int32_t* mask) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (x_idx == mask[i]) { x[i] = y[i]; }
  }
}
}  // namespace

template<typename T>
struct MaximumKernelUtil<DeviceType::kGPU, T> {
  static void CWiseMaxWithMask(DeviceCtx* ctx, const int64_t n, T* x, const T* y, const int y_idx,
                               int32_t* mask) {
    CWiseMaxWithMaskGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y,
                                                                                      y_idx, mask);
  }

  static void CWiseSetWithMask(DeviceCtx* ctx, const int64_t n, T* x, const T* y, const int x_idx,
                               const int32_t* mask) {
    CWiseSetWithMaskGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y,
                                                                                      x_idx, mask);
  }
};
#define INSTANTIATE_MAXIMUM_KERNEL_UTIL(type_cpp, type_proto) \
  template struct MaximumKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_MAXIMUM_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
