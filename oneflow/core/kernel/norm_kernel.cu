#include "oneflow/core/kernel/norm_kernel.h"
#include "oneflow/core/kernel/norm_kernel.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void AbsGpu(const int32_t n, const T epsilon, const T* in_dptr, T* abs_tmp_dptr) {
  CUDA_1D_KERNEL_LOOP(i, n) { abs_tmp_dptr[i] = std::abs(in_dptr[i]) + epsilon; }
}

template<typename T>
__global__ void L1NormBackwardGpu(const int32_t out_n, const int32_t offset, const T* out_diff_dptr,
                                  const T* in_dptr, T* in_diff_dptr) {
  for (int32_t i = blockIdx.x; i < out_n; i += gridDim.x) {
    for (int32_t j = threadIdx.x; j < offset; j += blockDim.x) {
      int32_t index = i * offset + j;
      in_diff_dptr[index] = L1NormInDiff(out_diff_dptr[i], in_dptr[index]);
    }
  }
}

}  // namespace

template<typename T>
struct NormKernelUtil<DeviceType::kGPU, T> {
  static void Abs(DeviceCtx* ctx, const int32_t n, const T epsilon, const T* in_dptr,
                  T* abs_tmp_dptr) {
    AbsGpu<<<n, kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, epsilon, in_dptr,
                                                                  abs_tmp_dptr);
  }

  static void L1NormBackward(DeviceCtx* ctx, const int32_t out_n, const int32_t offset,
                             const T* out_diff_dptr, const T* in_dptr, T* in_diff_dptr) {
    L1NormBackwardGpu<<<out_n * offset, kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        out_n, offset, out_diff_dptr, in_dptr, in_diff_dptr);
  }
};

#define INSTANTIATE_NORM_KERNEL_UTIL(type_cpp, type_proto) \
  template struct NormKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_NORM_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
