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
#include "oneflow/user/kernels/softmax_cross_entropy_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {
namespace user_op {

template<typename T>
struct CrossEntropyKernelUtil<DeviceType::kCPU, T> {
  static void ComputeEntropy(ep::Stream* stream, const int64_t num_instances,
                             const int64_t num_classes, const T* x, const T* labels, T* y) {
    FOR_RANGE(int64_t, i, 0, num_instances) {
      T tmp = 0;
      FOR_RANGE(int64_t, j, 0, num_classes) {
        T label = labels[i * num_classes + j];
        T prob = x[i * num_classes + j];
        // tmp -= label * SafeLog(prob);
        tmp -= label * logf((prob > 1e-20) ? prob : 1e-20);
      }
      y[i] = tmp;
    }
  }

  static void ComputeDiffWithSoftmax(ep::Stream* stream, const int64_t elem_cnt,
                                     const int64_t num_classes, const T* prob, const T* labels,
                                     const T* dy, T* dx) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) {
      const int32_t row_id = i / num_classes;
      dx[i] = dy[row_id] * (prob[i] - labels[i]);
    }
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SOFTMAX_CROSS_ENTROPY_KERNEL,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU), FLOATING_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU), FLOATING_DATA_TYPE_SEQ)
}  // namespace user_op
}  // namespace oneflow
