#include "oneflow/customized/kernels/two_stage_reduce_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename K>
__global__ void DivideMaxCountGpu(const int64_t n, const T* x, const K* max_count, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] / max_count[i]; }
}

template<typename T, typename K>
__global__ void ElemWiseSetWithMaskGpu(const int64_t n, const T* x, const K* mask, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = static_cast<bool>(mask[i]) ? x[i] : 0; }
}

template<typename T, typename K>
__global__ void MulCountGpu(const int64_t n, const T* x, const K* count, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] * count[i]; }
}

}  // namespace

template<typename T, typename K>
struct TwoStageReduceKernelUtil<DeviceType::kGPU, T, K> {
  static void DivideMaxCount(DeviceCtx* ctx, const int64_t n, const T* x, const K* max_count,
                             T* y) {
    DivideMaxCountGpu<T, K>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x,
                                                                                      max_count, y);
  }

  static void ElemWiseSetWithMask(DeviceCtx* ctx, const int64_t n, const T* x, const K* mask,
                                  T* y) {
    ElemWiseSetWithMaskGpu<T, K>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, mask,
                                                                                      y);
  }

  static void MulCount(DeviceCtx* ctx, const int64_t n, const T* x, const K* count, T* y) {
    MulCountGpu<T, K><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, x, count, y);
  }
};

#define INSTANTIATE_TWO_STAGE_REDUCE_KERNEL_UTIL_GPU(data_type_pair, index_type_pair)          \
  template struct TwoStageReduceKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                           OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_TWO_STAGE_REDUCE_KERNEL_UTIL_GPU,
                                 FLOATING_DATA_TYPE_SEQ INDEX_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ);
#undef INSTANTIATE_TWO_STAGE_REDUCE_KERNEL_UTIL_GPU

}  // namespace oneflow
