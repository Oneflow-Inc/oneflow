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
#ifndef ONEFLOW_USER_KERNELS_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_CUH_
#define ONEFLOW_USER_KERNELS_SPARSE_SOFTMAX__CROSS_ENTROPY_KERNEL_UTIL_CUH_

#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/user/kernels/sparse_softmax_cross_entropy_kernel_util.h"

namespace oneflow {
namespace user_op {

template<typename T, typename K>
__global__ void ComputeSparseSoftmaxCrossEntropyResultGpu(
    const int64_t num_instances, const int64_t num_classes, const int64_t depth,
    const int64_t lower_bound, const K* labels, T* sum_result, T* sub_result, T* out) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    assert(labels[i] >= 0);
    assert(labels[i] < depth);
    K label = labels[i] - lower_bound;
    if (label >= 0 && label < num_classes) {
      out[i] = SafeLog(sum_result[i]) - sub_result[i * num_classes + label];
    }
  }
}

template<typename K>
__global__ void ComputeSparseSoftmaxCrossEntropyResultGpuHalf(
    const int64_t num_instances, const int64_t num_classes, const int64_t depth,
    const int64_t lower_bound, const K* labels, half* sum_result, half* sub_result, half* out) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    assert(labels[i] >= 0);
    assert(labels[i] < depth);
    K label = labels[i] - lower_bound;
    if (label >= 0 && label < num_classes) {
      out[i] = __float2half(SafeLog(__half2float(sum_result[i]))
                            - __half2float(sub_result[i * num_classes + label]));
    }
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_CUH_
