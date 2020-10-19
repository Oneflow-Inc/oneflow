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
#include "oneflow/core/kernel/sparse_cross_entropy_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

template<typename T, typename K>
struct SparseCrossEntropyKernelUtil<DeviceType::kCPU, T, K> {
  static void ComputeEntropy(DeviceCtx* ctx, int64_t num_instances, int64_t num_classes, const T* x,
                             const K* labels, T* y) {
    FOR_RANGE(int64_t, i, 0, num_instances) {
      K label = labels[i];
      CHECK_GE(label, 0);
      CHECK_LT(label, num_classes);
      y[i] = -SafeLog(x[i * num_classes + label]);
    }
  }

  static void ComputeDiff(DeviceCtx* ctx, int64_t num_instances, int64_t num_classes, const T* x,
                          const K* labels, T* dx) {
    FOR_RANGE(int64_t, i, 0, num_instances) {
      K label = labels[i];
      CHECK_GE(label, 0);
      CHECK_LT(label, num_classes);
      dx[i * num_classes + label] = -1 / MaxWithLogThreshold(x[i * num_classes + label]);
    }
  }

  static void ComputeDiff(DeviceCtx* ctx, int64_t num_instances, int64_t num_classes, const T* x,
                          const K* labels, const T* dy, T* dx) {
    FOR_RANGE(int64_t, i, 0, num_instances) {
      K label = labels[i];
      CHECK_GE(label, 0);
      CHECK_LT(label, num_classes);
      dx[i * num_classes + label] = -dy[i] / MaxWithLogThreshold(x[i * num_classes + label]);
    }
  }
};

#define INSTANTIATE_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_CPU(data_type_pair, index_type_pair)          \
  template struct SparseCrossEntropyKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                               OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_CPU,
                                 FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ);
#undef INSTANTIATE_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_CPU

}  // namespace oneflow
