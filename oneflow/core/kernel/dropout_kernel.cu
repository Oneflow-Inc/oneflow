#include "oneflow/core/kernel/dropout_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void MaskAndScaleGpu(const int64_t n, float threshold, float scale, const T* x,
                                const float* random_mask, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] * (random_mask[i] > threshold) * scale; }
}

}  // namespace

template<typename T>
struct DropoutKernelUtil<DeviceType::kGPU, T> final {
  static void MaskAndScale(DeviceCtx* ctx, const int64_t n, float threshold, float scale,
                           const T* x, const float* random_mask, T* y) {
    MaskAndScaleGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, threshold, scale, x, random_mask, y);
  }
};

#define INITIATE_DROPOUT_KERNEL_UTIL(T, type_proto) \
  template struct DropoutKernelUtil<DeviceType::kGPU, T>;

OF_PP_FOR_EACH_TUPLE(INITIATE_DROPOUT_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
