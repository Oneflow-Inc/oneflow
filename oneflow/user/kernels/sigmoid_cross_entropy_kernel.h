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
Copyright 2021 The OneFlow Authors. All rights reserved.
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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace user_op {

template<DeviceType device_type, typename T>
struct CrossEntropyKernelUtil {
  static void ComputeEntropy(DeviceCtx* ctx, const int64_t n, const T* prediction, const T* label,
                             T* loss);
  static void ComputeDiffWithSigmoid(DeviceCtx* ctx, int64_t n, const T* prediction, const T* label,
                                     T* prediction_diff);
};

template<DeviceType device_type, typename T>
class SigmoidCrossEntropyKernel final : public user_op::OpKernel {
 public:
  SigmoidCrossEntropyKernel() = default;
  ~SigmoidCrossEntropyKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* prediction = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    user_op::Tensor* loss = ctx->Tensor4ArgNameAndIndex("loss", 0);
    const auto n = prediction->shape().elem_cnt();
    CrossEntropyKernelUtil<device_type, T>::ComputeEntropy(
        ctx->device_ctx(), n, prediction->dptr<T>(), label->dptr<T>(), loss->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(device_type_v, dtype_pair)                     \
  REGISTER_USER_KERNEL("sigmoid_cross_entropy")                                              \
      .SetCreateFn<SigmoidCrossEntropyKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device_type_v)                            \
                       & (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(dtype_pair)) \
                       & (user_op::HobDataType("loss", 0) == OF_PP_PAIR_SECOND(dtype_pair)));

template<DeviceType device_type, typename T>
class SigmoidCrossEntropyGradKernel final : public user_op::OpKernel {
 public:
  SigmoidCrossEntropyGradKernel() = default;
  ~SigmoidCrossEntropyGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    const user_op::Tensor* loss_diff = ctx->Tensor4ArgNameAndIndex("loss_diff", 0);
    const user_op::Tensor* prediction = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    user_op::Tensor* prediction_diff = ctx->Tensor4ArgNameAndIndex("prediction_diff", 0);
    const int64_t n = prediction->shape().elem_cnt();
    CrossEntropyKernelUtil<device_type, T>::ComputeDiffWithSigmoid(
        ctx->device_ctx(), n, prediction->dptr<T>(), label->dptr<T>(),
        prediction_diff->mut_dptr<T>());
    KernelUtil<device_type, T>::Mul(ctx->device_ctx(), n, prediction_diff->dptr<T>(),
                                    loss_diff->dptr<T>(), prediction_diff->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(device_type_v, dtype_pair)                    \
  REGISTER_USER_KERNEL("sigmoid_cross_entropy_grad")                                             \
      .SetCreateFn<SigmoidCrossEntropyGradKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>>() \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceTag() == device_type_v)                                             \
          & (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(dtype_pair))                  \
          & (user_op::HobDataType("prediction_diff", 0) == OF_PP_PAIR_SECOND(dtype_pair)));

}  // namespace user_op

}  // namespace oneflow
