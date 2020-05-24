#include "oneflow/customized/kernels/sparse_cross_entropy_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {
namespace user_op {

namespace {

template<typename T, typename K>
__global__ void ComputeEntropyGpu(const int64_t num_instances, const int64_t num_classes,
                                  const T* x, const K* labels, T* y) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    K label = labels[i];
    assert(label >= 0);
    assert(label < num_classes);
    y[i] = -SafeLog(x[i * num_classes + label]);
  }
}

template<typename K>
__global__ void ComputeEntropyGpuHalf(const int64_t num_instances, const int64_t num_classes,
                                      const half* x, const K* labels, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    K label = labels[i];
    assert(label >= 0);
    assert(label < num_classes);
    y[i] = __hneg(SafeLog<half>(x[i * num_classes + label]));
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T, typename K>
__global__ void ComputeDiffGpu(const int64_t num_instances, const int64_t num_classes, const T* x,
                               const K* labels, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    K label = labels[i];
    assert(label >= 0);
    assert(label < num_classes);
    dx[i * num_classes + label] = -dy[i] / MaxWithLogThreshold(x[i * num_classes + label]);
  }
}

template<typename K>
__global__ void ComputeDiffGpuHalf(const int64_t num_instances, const int64_t num_classes,
                                   const half* x, const K* labels, const half* dy, half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    K label = labels[i];
    assert(label >= 0);
    assert(label < num_classes);
    dx[i * num_classes + label] =
        __hneg(__hdiv(__float2half(dy[i]), MaxWithLogThreshold<half>(x[i * num_classes + label])));
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T, typename K>
__global__ void ComputeDiffWithSoftmaxGpu(const int64_t elem_cnt, const int64_t num_classes,
                                          const T* prob, const K* labels, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t row_id = i / num_classes;
    const int32_t col_id = i - row_id * num_classes;
    K label = labels[row_id];
    assert(label >= 0);
    assert(label < num_classes);
    if (label == col_id) {
      dx[i] = dy[row_id] * (prob[i] - 1);
    } else {
      dx[i] = dy[row_id] * prob[i];
    }
  }
}

template<typename K>
__global__ void ComputeDiffWithSoftmaxGpuHalf(const int64_t elem_cnt, const int64_t num_classes,
                                              const half* prob, const K* labels, const half* dy,
                                              half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t row_id = i / num_classes;
    const int32_t col_id = i - row_id * num_classes;
    K label = labels[row_id];
    assert(label >= 0);
    assert(label < num_classes);
    if (label == col_id) {
      dx[i] = __hmul(dy[row_id], __hsub(prob[i], __float2half(1.0)));
    } else {
      dx[i] = __hmul(dy[row_id], prob[i]);
    }
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

}  // namespace

template<typename T, typename K>
struct SparseCrossEntropyKernelUtil<DeviceType::kGPU, T, K> {
  static void ComputeEntropy(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                             const T* x, const K* labels, T* y) {
    ComputeEntropyGpu<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                        ctx->cuda_stream()>>>(num_instances, num_classes, x, labels, y);
  }

  static void ComputeDiff(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                          const T* x, const K* labels, const T* dy, T* dx) {
    ComputeDiffGpu<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                     ctx->cuda_stream()>>>(num_instances, num_classes, x, labels, dy, dx);
  }

  static void ComputeDiffWithSoftmax(DeviceCtx* ctx, const int64_t elem_cnt,
                                     const int64_t num_classes, const T* prob, const K* labels,
                                     const T* dy, T* dx) {
    ComputeDiffWithSoftmaxGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                ctx->cuda_stream()>>>(elem_cnt, num_classes, prob, labels, dy, dx);
  }
};

template<typename K>
struct SparseCrossEntropyKernelUtil<DeviceType::kGPU, float16, K> {
  static void ComputeEntropy(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                             const float16* x, const K* labels, float16* y) {
    ComputeEntropyGpuHalf<K>
        <<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            num_instances, num_classes, reinterpret_cast<const half*>(x), labels,
            reinterpret_cast<half*>(y));
  }

  static void ComputeDiff(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                          const float16* x, const K* labels, const float16* dy, float16* dx) {
    ComputeDiffGpuHalf<K>
        <<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            num_instances, num_classes, reinterpret_cast<const half*>(x), labels,
            reinterpret_cast<const half*>(dy), reinterpret_cast<half*>(dx));
  }

  static void ComputeDiffWithSoftmax(DeviceCtx* ctx, const int64_t elem_cnt,
                                     const int64_t num_classes, const float16* prob,
                                     const K* labels, const float16* dy, float16* dx) {
    ComputeDiffWithSoftmaxGpuHalf<K>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, num_classes, reinterpret_cast<const half*>(prob), labels,
            reinterpret_cast<const half*>(dy), reinterpret_cast<half*>(dx));
  }
};

#define INSTANTIATE_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_GPU(data_type_pair, index_type_pair)          \
  template struct SparseCrossEntropyKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                               OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_GPU,
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_GPU

}  // namespace user_op
}  // namespace oneflow
