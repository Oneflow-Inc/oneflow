#include "oneflow/core/kernel/multi_ring_all_reduce_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

namespace {

template<typename T>
__global__ void CopyGpu(const int64_t elem_cnt, T* dst, const T* src) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { dst[i] = src[i]; }
}

template<typename T>
__global__ void CopyGpu(const int64_t elem_cnt, T* dst0, T* dst1, const T* src) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T v = src[i];
    dst0[i] = v;
    dst1[i] = v;
  }
}

template<typename T>
__global__ void ReduceGpu(const int64_t elem_cnt, T* dst, const T* src0, const T* src1) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { dst[i] = src0[i] + src1[i]; }
}

template<typename T>
__global__ void ReduceGpu(const int64_t elem_cnt, T* dst0, T* dst1, const T* src0, const T* src1) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T v = src0[i] + src1[i];
    dst0[i] = v;
    dst1[i] = v;
  }
}

}  // namespace

template<typename T>
struct MultiRingAllReduceKernelUtil<DeviceType::kGPU, T> {
  static void Copy(DeviceCtx* ctx, T* dst, const T* src, int64_t size) {
    CopyGpu<T><<<BlocksNum4ThreadsNum(size), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        size, dst, src);
  }
  static void Copy(DeviceCtx* ctx, T* dst0, T* dst1, const T* src, int64_t size) {
    CopyGpu<T><<<BlocksNum4ThreadsNum(size), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        size, dst0, dst1, src);
  }
  static void Reduce(DeviceCtx* ctx, T* dst, const T* src0, const T* src1, int64_t size) {
    ReduceGpu<T><<<BlocksNum4ThreadsNum(size), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        size, dst, src0, src1);
  }
  static void Reduce(DeviceCtx* ctx, T* dst0, T* dst1, const T* src0, const T* src1, int64_t size) {
    ReduceGpu<T><<<BlocksNum4ThreadsNum(size), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        size, dst0, dst1, src0, src1);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template struct MultiRingAllReduceKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)
#undef INSTANTIATE_GPU_KERNEL_UTIL

}  // namespace oneflow
