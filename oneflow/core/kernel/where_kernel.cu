#include "oneflow/core/kernel/where_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void WhereGpu(const int64_t n, const T* cond_dptr, const T* x_dptr, const T* y_dptr,
                         T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out_dptr[i] = (cond_dptr[i] != 0) * x_dptr[i] + (cond_dptr[i] == 0) * y_dptr[i];
  }
}

template<typename T>
__global__ void CmptXDiffGpu(const int64_t n, const T* cond_dptr, const T* out_diff_dptr,
                             T* x_diff_dptr) {
  CUDA_1D_KERNEL_LOOP(i, n) { x_diff_dptr[i] = (cond_dptr[i] != 0) * out_diff_dptr[i]; }
}

template<typename T>
__global__ void CmptYDiffGpu(const int64_t n, const T* cond_dptr, const T* out_diff_dptr,
                             T* y_diff_dptr) {
  CUDA_1D_KERNEL_LOOP(i, n) { y_diff_dptr[i] = (cond_dptr[i] == 0) * out_diff_dptr[i]; }
}

}  // namespace

template<typename T>
struct WhereKernelUtil<DeviceType::kGPU, T> {
  static void Where(DeviceCtx* ctx, const int64_t n, const T* cond_dptr, const T* x_dptr,
                    const T* y_dptr, T* out_dptr) {
    WhereGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, cond_dptr, x_dptr, y_dptr, out_dptr);
  }
  static void CmptXDiff(DeviceCtx* ctx, const int64_t n, const T* cond_dptr, const T* out_diff_dptr,
                        T* x_diff_dptr) {
    CmptXDiffGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, cond_dptr, out_diff_dptr, x_diff_dptr);
  }
  static void CmptYDiff(DeviceCtx* ctx, const int64_t n, const T* cond_dptr, const T* out_diff_dptr,
                        T* y_diff_dptr) {
    CmptYDiffGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, cond_dptr, out_diff_dptr, y_diff_dptr);
  }
};

#define INSTANTIATE_WHERE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct WhereKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_WHERE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
