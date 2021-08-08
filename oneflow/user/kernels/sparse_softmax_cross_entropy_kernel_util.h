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

#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"

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

template<DeviceType device_type, typename T, typename K>
void SparseSoftmaxCrossEntropyComputePart(DeviceCtx* ctx, const int64_t num_instances,
                                          const int64_t num_classes, const T* in, T* prob,
                                          void* temp_storage, const size_t reduce_operation_size,
                                          T* sum_result, T* sub_result,
                                          const MemoryCase& prob_mem_case,
                                          const MemoryCase& tmp_buffer_mem_case) {
  auto Val = NdarrayUtil<device_type, T>::GetValNdarrayBuilder();
  auto Var = NdarrayUtil<device_type, T>::GetVarNdarrayBuilder();

  T* reduce_storage = reinterpret_cast<T*>(temp_storage);

  auto reduce_storage_var =
      Var({static_cast<int64_t>(reduce_operation_size / sizeof(T))}, reduce_storage);

  // max | sum_result[i] = Max_j(in[i][j])
  NdarrayUtil<device_type, T>::ReduceMax(ctx, Var({num_instances, 1}, sum_result),
                                         Val({num_instances, num_classes}, in), reduce_storage_var);
  // sub | sub_result[i][j] = in[i][j] - sum_result[i]
  NdarrayUtil<device_type, T>::BroadcastSub(ctx, Var({num_instances, num_classes}, sub_result),
                                            Val({num_instances, num_classes}, in),
                                            Val({num_instances, 1}, sum_result));
  // copy | sub_result => prob
  AutoMemcpy(ctx, prob, sub_result, reduce_operation_size, prob_mem_case, tmp_buffer_mem_case);
  // exp | prob[i][j] = exp(prob[i][j])
  NdarrayUtil<device_type, T>::InplaceExp(ctx, Var({num_instances, num_classes}, prob));
  // sum | sum_result[i] = Sum_j(prob[i][j])
  NdarrayUtil<device_type, T>::ReduceSum(ctx, Var({num_instances, 1}, sum_result),
                                         Val({num_instances, num_classes}, prob),
                                         reduce_storage_var);

  NdarrayUtil<device_type, T>::InplaceBroadcastDiv(
      ctx, Var({num_instances, num_classes}, prob),
      Val({num_instances, 1}, sum_result));  // for backward
}
}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_H_
