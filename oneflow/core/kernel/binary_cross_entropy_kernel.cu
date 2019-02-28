#include "oneflow/core/kernel/binary_cross_entropy_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T, typename K>
__global__ void ComputeEntropyGpu(int64_t num_instances, const T* x, const K* labels, T* y) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    const K label = labels[i];
    assert(label == 0 || label == 1);
    y[i] = -SafeLog(label == 0 ? OneVal<T>::value - x[i] : x[i]);
  }
}

template<typename T, typename K>
__global__ void ComputeDiffGpu(int64_t num_instances, const T* x, const K* labels, const T* dy,
                               T* dx) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    const K label = labels[i];
    assert(label == 0 || label == 1);
    dx[i] = -dy[i] / MaxWithLogThreshold(label == 0 ? OneVal<T>::value - x[i] : x[i]);
  }
}

}  // namespace

template<typename T, typename K>
struct BinaryCrossEntropyKernelUtil<DeviceType::kGPU, T, K> {
  static void ComputeEntropy(DeviceCtx* ctx, int64_t num_instances, const T* x, const K* labels,
                             T* y) {
    ComputeEntropyGpu<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                        ctx->cuda_stream()>>>(num_instances, x, labels, y);
  }

  static void ComputeDiff(DeviceCtx* ctx, int64_t num_instances, const T* x, const K* labels,
                          const T* dy, T* dx) {
    ComputeDiffGpu<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                     ctx->cuda_stream()>>>(num_instances, x, labels, dy, dx);
  }
};

#define INSTANTIATE_BINARY_CROSS_ENTROPY_KERNEL_UTIL_GPU(data_type_pair, index_type_pair)          \
  template struct BinaryCrossEntropyKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                               OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BINARY_CROSS_ENTROPY_KERNEL_UTIL_GPU,
                                 FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ);
#undef INSTANTIATE_BINARY_CROSS_ENTROPY_KERNEL_UTIL_GPU

}  // namespace oneflow
