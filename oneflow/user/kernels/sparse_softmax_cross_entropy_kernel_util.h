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
#ifndef ONEFLOW_USER_KERNELS_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_SPARSE_SOFTMAX__CROSS_ENTROPY_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace user_op {

template<typename T>
size_t SparseSoftmaxCrossEntropySubResultSize(int64_t num_instances, int64_t num_classes) {
  return GetCudaAlignedSize(num_instances * num_classes * sizeof(T));
}

template<typename T>
size_t SparseSoftmaxCrossEntropySumResultSize(int64_t num_instances) {
  return GetCudaAlignedSize(num_instances * sizeof(T));
}

template<typename T>
size_t SparseSoftmaxCrossEntropyReduceOperationSize(int64_t num_instances, int64_t num_classes) {
  return GetCudaAlignedSize(num_instances * num_classes * sizeof(T));
}

template<typename T>
size_t SparseSoftmaxCrossEntropyTempStorageSize(int64_t num_instances, int64_t num_classes) {
  return SparseSoftmaxCrossEntropyReduceOperationSize<T>(num_instances, num_classes)
         + SparseSoftmaxCrossEntropySubResultSize<T>(num_instances, num_classes)
         + SparseSoftmaxCrossEntropySumResultSize<T>(num_instances);
}
template<DeviceType device_type, typename T, typename K>
struct SparseSoftmaxCrossEntropyKernelUtil {
  static void Compute(DeviceCtx* ctx, const int64_t n, const int64_t w, const int64_t depth,
                      const int64_t lower_bound, const T* in, T* prob, const K* labels, T* y,
                      void* temp_storage, const size_t temp_storage_bytes,
                      const MemoryCase& prob_mem_case, const MemoryCase& tmp_buffer_mem_case);
  static void ComputeDiff(DeviceCtx* ctx, const int64_t elem_cnt, const int64_t num_classes,
                          const int64_t depth, const int64_t lower_bound, const T* prob,
                          const K* labels, const T* dy, T* dx);
};
}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_H_
