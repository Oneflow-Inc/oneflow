#include "oneflow/core/kernel/random_mask_like_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

__global__ void GenMaskGpu(const int64_t n, float threshold, const float* random_tmp,
                           int8_t* mask) {
  CUDA_1D_KERNEL_LOOP(i, n) { mask[i] = random_tmp[i] > threshold; }
}

}  // namespace

template<>
void RandomMaskLikeKernelUtil<DeviceType::kGPU>::GenMask(DeviceCtx* ctx, const int64_t n,
                                                         float threshold, const float* random_tmp,
                                                         int8_t* mask) {
  GenMaskGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, threshold, random_tmp, mask);
}

template struct RandomMaskLikeKernelUtil<DeviceType::kGPU>;

}  // namespace oneflow
