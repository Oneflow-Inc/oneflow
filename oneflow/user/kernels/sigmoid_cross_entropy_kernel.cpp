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
#include "oneflow/user/kernels/sigmoid_cross_entropy_kernel.h"

namespace oneflow {
namespace user_op {

template<typename T>
struct CrossEntropyKernelUtil<DeviceType::kCPU, T> {
  static void ComputeEntropy(DeviceCtx* ctx, const int64_t n, const T* prediction, const T* label,
                             T* loss) {
    FOR_RANGE(int64_t, index, 0, n) {
      loss[index] =
          -1.f * prediction[index] * (label[index] - (prediction[index] >= 0))
          + logf(1 + expf(prediction[index] - 2 * prediction[index] * (prediction[index] >= 0)));
    }
  }

  static void ComputeDiffWithSigmoid(DeviceCtx* ctx, int64_t n, const T* prediction, const T* label,
                                     T* prediction_diff) {
    FOR_RANGE(int64_t, index, 0, n) {
      prediction_diff[index] = 1.f / (1.f + expf(-prediction[index])) - label[index];
    }
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU), FLOATING_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU), FLOATING_DATA_TYPE_SEQ)

}  // namespace user_op
}  // namespace oneflow
