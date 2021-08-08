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

namespace oneflow {
namespace user_op {

template<typename T, typename K>
struct SparseSoftmaxCrossEntropyKernelUtil<DeviceType::kCPU, T, K> {
  static void Compute(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                      const int64_t depth, const int64_t lower_bound, const T* in, T* prob,
                      const K* labels, T* y, void* temp_storage, const size_t temp_storage_bytes,
                      const MemoryCase& prob_mem_case, const MemoryCase& tmp_buffer_mem_case) {
    const size_t min_temp_storage_bytes =
        SparseSoftmaxCrossEntropyTempStorageSize<T>(num_instances, num_classes);
    CHECK_GE(temp_storage_bytes, min_temp_storage_bytes);

    const size_t reduce_operation_size =
        SparseSoftmaxCrossEntropyReduceOperationSize<T>(num_instances, num_classes);
    const size_t sum_result_size = SparseSoftmaxCrossEntropySumResultSize<T>(num_instances);

    T* sum_result = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                         + reduce_operation_size);
    T* sub_result = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                         + reduce_operation_size + sum_result_size);

    SparseSoftmaxCrossEntropyComputePart<DeviceType::kCPU, T, K>(
        ctx, num_instances, num_classes, in, prob, temp_storage, reduce_operation_size, sum_result,
        sub_result, prob_mem_case, tmp_buffer_mem_case);
    FOR_RANGE(int64_t, i, 0, num_instances) {
      CHECK_GE(labels[i], 0);
      CHECK_LT(labels[i], depth);
      K label = labels[i] - lower_bound;
      if (label >= 0 && label < num_classes) {
        y[i] = SafeLog(sum_result[i]) - sub_result[i * num_classes + label];
      }
    }
  }

  static void ComputeDiff(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                          const int64_t depth, const int64_t lower_bound, const T* prob,
                          const K* labels, const T* dy, T* dx) {
    FOR_RANGE(int64_t, i, 0, num_instances) {
      const int32_t row_id = i / num_classes;
      const int32_t col_id = i - row_id * num_classes;
      CHECK_GE(labels[row_id], 0);
      CHECK_LT(labels[row_id], depth);
      K label = labels[row_id] - lower_bound;

      if (label == col_id) {
        dx[i] = dy[row_id] * (prob[i] - 1);
      } else {
        dx[i] = dy[row_id] * prob[i];
      }
    }
  }
};
#define INSTANTIATE_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_CPU(data_type_pair, index_type_pair) \
  template struct SparseSoftmaxCrossEntropyKernelUtil<                                            \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_CPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_CPU
}  // namespace user_op
}  // namespace oneflow
