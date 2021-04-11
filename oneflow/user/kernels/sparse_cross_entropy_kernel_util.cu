/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/user/kernels/sparse_cross_entropy_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {
namespace user_op {

namespace {

template<typename T, typename K>
__global__ void ComputeEntropyGpu(const int64_t num_instances, const int64_t num_classes,
                                  const int64_t depth, const int64_t lower_bound, const T* x,
                                  const K* labels, T* y) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    assert(labels[i] >= 0);
    assert(labels[i] < depth);
    K label = labels[i] - lower_bound;
    if (label >= 0 && label < num_classes) { y[i] = -SafeLog(x[i * num_classes + label]); }
  }
}

template<typename K>
__global__ void ComputeEntropyGpuHalf(const int64_t num_instances, const int64_t num_classes,
                                      const int64_t depth, const int64_t lower_bound, const half* x,
                                      const K* labels, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    assert(labels[i] >= 0);
    assert(labels[i] < depth);
    K label = labels[i] - lower_bound;
    if (label >= 0 && label < num_classes) {
      y[i] = __float2half(-SafeLog(__half2float(x[i * num_classes + label])));
    }
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T, typename K>
__global__ void ComputeDiffGpu(const int64_t num_instances, const int64_t num_classes,
                               const int64_t depth, const int64_t lower_bound, const T* x,
                               const K* labels, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    assert(labels[i] >= 0);
    assert(labels[i] < depth);
    K label = labels[i] - lower_bound;
    if (label >= 0 && label < num_classes) {
      dx[i * num_classes + label] = -dy[i] / MaxWithLogThreshold(x[i * num_classes + label]);
    }
  }
}

template<typename K>
__global__ void ComputeDiffGpuHalf(const int64_t num_instances, const int64_t num_classes,
                                   const int64_t depth, const int64_t lower_bound, const half* x,
                                   const K* labels, const half* dy, half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    assert(labels[i] >= 0);
    assert(labels[i] < depth);
    K label = labels[i] - lower_bound;
    if (label >= 0 && label < num_classes) {
      dx[i * num_classes + label] =
          __hneg(__hdiv(__float2half(dy[i]), MaxWithLogThreshold(x[i * num_classes + label])));
    }
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T, typename K>
__global__ void ComputeDiffWithSoftmaxGpu(const int64_t elem_cnt, const int64_t num_classes,
                                          const int64_t depth, const int64_t lower_bound,
                                          const T* prob, const K* labels, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t row_id = i / num_classes;
    const int32_t col_id = i - row_id * num_classes;
    assert(labels[row_id] >= 0);
    assert(labels[row_id] < depth);
    K label = labels[row_id] - lower_bound;
    if (label == col_id) {
      dx[i] = dy[row_id] * (prob[i] - 1);
    } else {
      dx[i] = dy[row_id] * prob[i];
    }
  }
}

template<typename K>
__global__ void ComputeDiffWithSoftmaxGpuHalf(const int64_t elem_cnt, const int64_t num_classes,
                                              const int64_t depth, const int64_t lower_bound,
                                              const half* prob, const K* labels, const half* dy,
                                              half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t row_id = i / num_classes;
    const int32_t col_id = i - row_id * num_classes;
    assert(labels[row_id] >= 0);
    assert(labels[row_id] < depth);
    K label = labels[row_id] - lower_bound;
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

template<typename K>
__global__ void ComputeDiffWithSoftmaxGpuHalf2(const int64_t elem_cnt, const int64_t num_classes,
                                               const int64_t depth, const int64_t lower_bound,
                                               const half* prob, const K* labels, const half* dy,
                                               half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  const int64_t h2_num_classes = num_classes / 2;
  const int64_t h2_elem_cnt = elem_cnt / 2;
  const auto* prob_h2 = reinterpret_cast<const half2*>(prob);
  auto* dx_h2 = reinterpret_cast<half2*>(dx);
  CUDA_1D_KERNEL_LOOP(i, h2_elem_cnt) {
    const int32_t row_id = i / h2_num_classes;
    const int32_t h2_col_id = i - row_id * h2_num_classes;
    assert(labels[row_id] >= 0);
    assert(labels[row_id] < depth);
    K label = labels[row_id] - lower_bound;
    const half2 prob_h2_i = prob_h2[i];
    const half dy_row = dy[row_id];
    half2 dx_h2_i;
    dx_h2_i.x = __hmul(dy_row, __hsub(prob_h2_i.x, static_cast<half>(label == 2 * h2_col_id)));
    dx_h2_i.y = __hmul(dy_row, __hsub(prob_h2_i.y, static_cast<half>(label == 2 * h2_col_id + 1)));
    dx_h2[i] = dx_h2_i;
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
                             const int64_t depth, const int64_t lower_bound, const T* x,
                             const K* labels, T* y) {
    ComputeEntropyGpu<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                        ctx->cuda_stream()>>>(num_instances, num_classes, depth, lower_bound, x,
                                              labels, y);
  }

  static void ComputeDiff(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                          const int64_t depth, const int64_t lower_bound, const T* x,
                          const K* labels, const T* dy, T* dx) {
    ComputeDiffGpu<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                     ctx->cuda_stream()>>>(num_instances, num_classes, depth, lower_bound, x,
                                           labels, dy, dx);
  }

  static void ComputeDiffWithSoftmax(DeviceCtx* ctx, const int64_t elem_cnt,
                                     const int64_t num_classes, const int64_t depth,
                                     const int64_t lower_bound, const T* prob, const K* labels,
                                     const T* dy, T* dx) {
    ComputeDiffWithSoftmaxGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                ctx->cuda_stream()>>>(elem_cnt, num_classes, depth, lower_bound,
                                                      prob, labels, dy, dx);
  }
};

template<typename K>
struct SparseCrossEntropyKernelUtil<DeviceType::kGPU, float16, K> {
  static void ComputeEntropy(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                             const int64_t depth, const int64_t lower_bound, const float16* x,
                             const K* labels, float16* y) {
    ComputeEntropyGpuHalf<K>
        <<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            num_instances, num_classes, depth, lower_bound, reinterpret_cast<const half*>(x),
            labels, reinterpret_cast<half*>(y));
  }

  static void ComputeDiff(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                          const int64_t depth, const int64_t lower_bound, const float16* x,
                          const K* labels, const float16* dy, float16* dx) {
    ComputeDiffGpuHalf<K>
        <<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            num_instances, num_classes, depth, lower_bound, reinterpret_cast<const half*>(x),
            labels, reinterpret_cast<const half*>(dy), reinterpret_cast<half*>(dx));
  }

  static void ComputeDiffWithSoftmax(DeviceCtx* ctx, const int64_t elem_cnt,
                                     const int64_t num_classes, const int64_t depth,
                                     const int64_t lower_bound, const float16* prob,
                                     const K* labels, const float16* dy, float16* dx) {
    if (num_classes % 2 == 0) {
      ComputeDiffWithSoftmaxGpuHalf2<K>
          <<<BlocksNum4ThreadsNum(elem_cnt / 2), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              elem_cnt, num_classes, depth, lower_bound, reinterpret_cast<const half*>(prob),
              labels, reinterpret_cast<const half*>(dy), reinterpret_cast<half*>(dx));
    } else {
      ComputeDiffWithSoftmaxGpuHalf<K>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              elem_cnt, num_classes, depth, lower_bound, reinterpret_cast<const half*>(prob),
              labels, reinterpret_cast<const half*>(dy), reinterpret_cast<half*>(dx));
    }
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
