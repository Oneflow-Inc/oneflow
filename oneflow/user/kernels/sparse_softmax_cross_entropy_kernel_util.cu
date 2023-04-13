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
#include "oneflow/user/kernels/sparse_softmax_cross_entropy_kernel_util.h"
#include "oneflow/core/cuda/softmax.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace user_op {
namespace {

template<typename T>
__inline__ __device__ T Exp(T x);

template<>
__inline__ __device__ float Exp<float>(float x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __expf(x);
#else
  return exp(x);
#endif
}

template<>
__inline__ __device__ double Exp<double>(double x) {
  return exp(x);
}

template<>
__inline__ __device__ half Exp<half>(half x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __float2half(__expf(__half2float(x)));
#else
  return __float2half(exp(__half2float(x)));
#endif
}

template<typename T, typename K, typename IndexType>
__global__ void ComputeDiffGpu(const int64_t num_instances, const int64_t num_classes,
                               const int64_t depth, const int64_t lower_bound, const T* prob,
                               const K* labels, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP_T(IndexType, i, num_instances) {
    const IndexType row_id = i / num_classes;
    const IndexType col_id = i - row_id * num_classes;
    assert(labels[row_id] >= 0);
    assert(labels[row_id] < depth);
    K label = labels[row_id] - lower_bound;
    if (label == col_id) {
      dx[i] = dy[row_id] * (Exp(prob[i]) - 1);
    } else {
      dx[i] = dy[row_id] * Exp(prob[i]);
    }
  }
}

template<typename K, typename IndexType>
__global__ void ComputeDiffGpuHalf(const int64_t num_instances, const int64_t num_classes,
                                   const int64_t depth, const int64_t lower_bound, const half* prob,
                                   const K* labels, const half* dy, half* dx) {
  CUDA_1D_KERNEL_LOOP_T(IndexType, i, num_instances) {
    const IndexType row_id = i / num_classes;
    const IndexType col_id = i - row_id * num_classes;
    assert(labels[row_id] >= 0);
    assert(labels[row_id] < depth);
    K label = labels[row_id] - lower_bound;
    if (label == col_id) {
      dx[i] = __hmul(dy[row_id], __hsub(Exp(prob[i]), __float2half(1.0)));
    } else {
      dx[i] = __hmul(dy[row_id], Exp(prob[i]));
    }
  }
}

}  // namespace

template<typename T, typename K>
struct SparseSoftmaxCrossEntropyKernelUtil<DeviceType::kCUDA, T, K> {
  static void ComputeDiff(ep::Stream* stream, const int64_t num_instances,
                          const int64_t num_classes, const int64_t depth, const int64_t lower_bound,
                          const T* prob, const K* labels, const T* dy, T* dx) {
    if (num_instances < GetMaxVal<int32_t>() / 2) {
      ComputeDiffGpu<T, K, int32_t><<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock,
                                      0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          num_instances, num_classes, depth, lower_bound, prob, labels, dy, dx);
    } else {
      // NOTE(chengcheng): int division ('/') of i will reduce performance of int64_t.
      ComputeDiffGpu<T, K, int64_t><<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock,
                                      0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          num_instances, num_classes, depth, lower_bound, prob, labels, dy, dx);
    }
  }
};

template<typename K>
struct SparseSoftmaxCrossEntropyKernelUtil<DeviceType::kCUDA, float16, K> {
  static void ComputeDiff(ep::Stream* stream, const int64_t num_instances,
                          const int64_t num_classes, const int64_t depth, const int64_t lower_bound,
                          const float16* prob, const K* labels, const float16* dy, float16* dx) {
    if (num_instances < GetMaxVal<int32_t>() / 2) {
      ComputeDiffGpuHalf<K, int32_t><<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock,
                                       0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          num_instances, num_classes, depth, lower_bound, reinterpret_cast<const half*>(prob),
          labels, reinterpret_cast<const half*>(dy), reinterpret_cast<half*>(dx));
    } else {
      ComputeDiffGpuHalf<K, int64_t><<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock,
                                       0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          num_instances, num_classes, depth, lower_bound, reinterpret_cast<const half*>(prob),
          labels, reinterpret_cast<const half*>(dy), reinterpret_cast<half*>(dx));
    }
  }
};

#define INSTANTIATE_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_CUDA(data_type_pair, index_type_pair) \
  template struct SparseSoftmaxCrossEntropyKernelUtil<                                             \
      DeviceType::kCUDA, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_CUDA,
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_CUDA

}  // namespace user_op
}  // namespace oneflow
