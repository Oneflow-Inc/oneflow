#include "oneflow/core/kernel/add_kernel.h"

namespace oneflow {

namespace {

__global__ void half_gpu_add(const int64_t n, half* out, const half* in_0, const half* in_1) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = __hadd(in_0[i], in_1[i]); }
}

__global__ void half_gpu_add(const int64_t n, half* out, const half* in_0, const half* in_1,
                             const half* in_2) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = __hadd(in_0[i], __hadd(in_1[i], in_2[i])); }
}

__global__ void half_gpu_add(const int64_t n, half* out, const half* in_0, const half* in_1,
                             const half* in_2, const half* in_3) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = __hadd(in_0[i], __hadd(in_1[i], __hadd(in_2[i], in_3[i]))); }
}

}  // namespace

void HalfGpuAdd(DeviceCtx* ctx, const int64_t n, float16* out_dptr,
                const std::vector<const float16*>& in_dptrs) {
  half* half_out_dptr = reinterpret_cast<half*>(out_dptr);
  std::vector<const half*> half_in_dptrs;
  for (const float16* ptr : in_dptrs) {
    half_in_dptrs.push_back(reinterpret_cast<const half*>(ptr));
  }
  switch (half_in_dptrs.size()) {
    case 2:
      half_gpu_add<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, half_out_dptr, half_in_dptrs.at(0), half_in_dptrs.at(1));
      break;
    case 3:
      half_gpu_add<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, half_out_dptr, half_in_dptrs.at(0), half_in_dptrs.at(1), half_in_dptrs.at(2));
      break;
    case 4:
      half_gpu_add<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, half_out_dptr, half_in_dptrs.at(0), half_in_dptrs.at(1), half_in_dptrs.at(2),
          half_in_dptrs.at(3));
      break;
    default: LOG(FATAL) << "error in_dptrs size " << half_in_dptrs.size();
  }
}

}  // namespace oneflow
