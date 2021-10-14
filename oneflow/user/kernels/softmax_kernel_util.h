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
#ifndef ONEFLOW_USER_KERNELS_SOFTMAX_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_SOFTMAX_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

enum class SoftmaxAlgorithm {
  kSoftmax = 0,
  kLogSoftmax = 1,
};

template<typename T>
size_t SoftmaxReduceResultStorageSize(int64_t n) {
  return GetCudaAlignedSize(n * sizeof(T));
}

template<typename T>
size_t SoftmaxReduceOperationStorageSize(int64_t n, int64_t w) {
  return GetCudaAlignedSize(n * w * sizeof(T));
}

template<typename T, SoftmaxAlgorithm softmax_algorithm>
typename std::enable_if<softmax_algorithm == SoftmaxAlgorithm::kSoftmax, size_t>::type
SoftmaxComputeProbTempStorageSize(int64_t n, int64_t w) {
  return SoftmaxReduceResultStorageSize<T>(n) + SoftmaxReduceOperationStorageSize<T>(n, w);
}

template<typename T, SoftmaxAlgorithm softmax_algorithm>
typename std::enable_if<softmax_algorithm == SoftmaxAlgorithm::kLogSoftmax, size_t>::type
SoftmaxComputeProbTempStorageSize(int64_t n, int64_t w) {
  return SoftmaxReduceResultStorageSize<T>(n) + SoftmaxReduceOperationStorageSize<T>(n, w) * 2;
}

template<typename T, SoftmaxAlgorithm softmax_algorithm>
typename std::enable_if<softmax_algorithm == SoftmaxAlgorithm::kSoftmax, size_t>::type
SoftmaxComputeDiffTempStorageSize(int64_t n, int64_t w) {
  return SoftmaxReduceResultStorageSize<T>(n) + SoftmaxReduceOperationStorageSize<T>(n, w);
}

template<typename T, SoftmaxAlgorithm softmax_algorithm>
typename std::enable_if<softmax_algorithm == SoftmaxAlgorithm::kLogSoftmax, size_t>::type
SoftmaxComputeDiffTempStorageSize(int64_t n, int64_t w) {
  return SoftmaxReduceResultStorageSize<T>(n) + SoftmaxReduceOperationStorageSize<T>(n, w) * 2;
}

template<DeviceType device_type, SoftmaxAlgorithm softmax_algorithm, typename T>
struct SoftmaxKernelUtil {
  static void ComputeProb(DeviceCtx* ctx, int64_t n, int64_t w, const T* in, T* prob,
                          void* temp_storage, size_t temp_storage_bytes);
  static void ComputeDiff(DeviceCtx* ctx, int64_t n, int64_t w, const T* dy, const T* out, T* dx,
                          void* temp_storage, size_t temp_storage_bytes);
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_SOFTMAX_KERNEL_UTIL_H_
