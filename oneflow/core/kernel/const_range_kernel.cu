#include "oneflow/core/kernel/const_range_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void RangeFill(T start, int64_t size, T stride, T* out) {
  CUDA_1D_KERNEL_LOOP(i, size) { out[i] = start + stride * i; }
}

}  // namespace

template<typename T>
struct ConstRangeKernelUtil<DeviceType::kGPU, T> final {
  static void Fill(DeviceCtx* ctx, T start, int64_t size, T stride, T* out);
};

template<typename T>
void ConstRangeKernelUtil<DeviceType::kGPU, T>::Fill(DeviceCtx* ctx, T start, int64_t size,
                                                     T stride, T* out) {
  RangeFill<<<BlocksNum4ThreadsNum(size), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      start, size, stride, out);
}

#define INITIATE_CONST_RANGE_KERNEL_UTIL(T, type_proto) \
  template struct ConstRangeKernelUtil<DeviceType::kGPU, T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_CONST_RANGE_KERNEL_UTIL, INT_DATA_TYPE_SEQ);
#undef INITIATE_CONST_RANGE_KERNEL_UTIL

}  // namespace oneflow
