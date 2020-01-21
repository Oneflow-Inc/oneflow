#include "oneflow/core/kernel/square_sum_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

template<typename T>
__global__ void SquareSumGpu(int64_t n, const T* x, T* y) {
  T t_sum = 0;
  CUDA_1D_KERNEL_LOOP(i, n) {
    T x_i = x[i];
    t_sum += x_i * x_i;
  }
  typedef cub::BlockReduce<T, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T sum = BlockReduce(temp_storage).Sum(t_sum);
  if (threadIdx.x == 0) { gpu_atomic_add<T>(y, sum); }
}

}  // namespace

template<typename T>
struct SquareSumKernelUtil<DeviceType::kGPU, T> {
  static void SquareSum(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
        Memset<DeviceType::kGPU>(ctx, y, 0, sizeof(T));
        SquareSumGpu<T>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x,
            y);
  }
};

#define INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_GPU(type_cpp, type_proto) \
  template struct SquareSumKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_GPU, FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_GPU

}  // namespace oneflow
