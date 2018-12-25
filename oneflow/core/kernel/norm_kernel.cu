#include "oneflow/core/kernel/norm_kernel.h"
#include "oneflow/core/kernel/norm_kernel.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void AbsGpu(const int32_t n, const T epsilon, const T* in_dptr, T* abs_tmp_dptr) {
  CUDA_1D_KERNEL_LOOP(i, n) { abs_tmp_dptr[i] = std::abs(in_dptr[i]) + epsilon; }
}

template<typename T>
__global__ void L1NormBackwardGpu(const int32_t n, const int32_t offset, const T* out_diff_dptr,
                                  const T* in_dptr, T* in_diff_dptr) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    in_diff_dptr[i] = L1NormInDiff(out_diff_dptr[i / offset], in_dptr[i]);
  }
}

}  // namespace

template<typename T>
struct NormKernelUtil<DeviceType::kGPU, T> {
  static void Abs(DeviceCtx* ctx, const int32_t n, const T epsilon, const T* in_dptr,
                  T* abs_tmp_dptr) {
    AbsGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, epsilon, in_dptr, abs_tmp_dptr);
  }

  static void L1NormBackward(DeviceCtx* ctx, const int32_t in_n, const int32_t offset,
                             const T* out_diff_dptr, const T* in_dptr, T* in_diff_dptr) {
    L1NormBackwardGpu<<<BlocksNum4ThreadsNum(in_n), kCudaThreadsNumPerBlock, 0,
                        ctx->cuda_stream()>>>(in_n, offset, out_diff_dptr, in_dptr, in_diff_dptr);
  }
};

#define INSTANTIATE_NORM_KERNEL_UTIL(type_cpp, type_proto) \
  template struct NormKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_NORM_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
