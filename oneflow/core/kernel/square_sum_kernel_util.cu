#include "oneflow/core/kernel/square_sum_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

template<typename T, bool ONE_BLOCK>
__global__ void SquareSumGpu(int64_t n, const T* x, T* y) {
  T t_sum = 0;
  CUDA_1D_KERNEL_LOOP(i, n) { t_sum += x[i] * x[i]; }
  typedef cub::BlockReduce<T, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T b_sum = BlockReduce(temp_storage).Sum(t_sum);
  if (threadIdx.x == 0) {
    if (ONE_BLOCK) {
      *y = b_sum;
    } else {
      gpu_atomic_add<T>(y, b_sum);
    }
  }
}

}  // namespace

template<typename T>
struct SquareSumKernelUtil<DeviceType::kGPU, T> {
  static void SquareSum(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
    const int32_t num_blocks = BlocksNum4ThreadsNum(n);
    CHECK_GE(num_blocks, 0);
    if (num_blocks == 0) {
      Memset<DeviceType::kGPU>(ctx, y, 0, sizeof(T));
    } else if (num_blocks == 1) {
      SquareSumGpu<T, true><<<1, kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
    } else {
      Memset<DeviceType::kGPU>(ctx, y, 0, sizeof(T));
      SquareSumGpu<T, false>
          <<<num_blocks, kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
    }
  }
};

#define INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_GPU(type_cpp, type_proto) \
  template struct SquareSumKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_GPU, FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_GPU

}  // namespace oneflow
