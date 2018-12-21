#include "oneflow/core/kernel/sparse_cross_entropy_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T, typename K>
__global__ void ComputeEntropyGpu(int64_t num_instances, int64_t num_classes, const T* x,
                                  const K* labels, T* y) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    K label = labels[i];
    assert(label >= 0);
    assert(label < num_classes);
    y[i] = -SafeLog(x[i * num_classes + label]);
  }
}

template<typename T, typename K>
__global__ void ComputeDiffGpu(int64_t num_instances, int64_t num_classes, const T* x,
                               const K* labels, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    K label = labels[i];
    assert(label >= 0);
    assert(label < num_classes);
    dx[i * num_classes + label] = -1 / MaxWithLogThreshold(x[i * num_classes + label]);
  }
}

template<typename T, typename K>
__global__ void ComputeDiffGpu(int64_t num_instances, int64_t num_classes, const T* x,
                               const K* labels, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    K label = labels[i];
    assert(label >= 0);
    assert(label < num_classes);
    dx[i * num_classes + label] = -dy[i] / MaxWithLogThreshold(x[i * num_classes + label]);
  }
}

}  // namespace

template<typename T, typename K>
struct SparseCrossEntropyKernelUtil<DeviceType::kGPU, T, K> {
  static void ComputeEntropy(DeviceCtx* ctx, int64_t num_instances, int64_t num_classes, const T* x,
                             const K* labels, T* y) {
    ComputeEntropyGpu<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                        ctx->cuda_stream()>>>(num_instances, num_classes, x, labels, y);
  }

  static void ComputeDiff(DeviceCtx* ctx, int64_t num_instances, int64_t num_classes, const T* x,
                          const K* labels, T* dx) {
    ComputeDiffGpu<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                     ctx->cuda_stream()>>>(num_instances, num_classes, x, labels, dx);
  }

  static void ComputeDiff(DeviceCtx* ctx, int64_t num_instances, int64_t num_classes, const T* x,
                          const K* labels, const T* dy, T* dx) {
    ComputeDiffGpu<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                     ctx->cuda_stream()>>>(num_instances, num_classes, x, labels, dy, dx);
  }
};

#define INSTANTIATE_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_GPU(data_type_pair, index_type_pair)          \
  template struct SparseCrossEntropyKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                               OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_GPU,
                                 FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ);
#undef INSTANTIATE_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_GPU

}  // namespace oneflow
