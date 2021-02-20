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
#ifndef ONEFLOW_USER_KERNELS_SIGMOID_CROSS_ENTROPY_KERNEL_H_
#define ONEFLOW_USER_KERNELS_SIGMOID_CROSS_ENTROPY_KERNEL_H_
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/math_unary_elementwise_func.h"

namespace oneflow {

template<typename PredT, typename LabelT>
struct SigmoidCrossEntropyFunctor {
  OF_DEVICE_FUNC PredT operator()(const PredT prediction, const LabelT label) const {
    return -1.f * prediction * (label - (prediction >= 0))
           + LogFunctor<PredT>::Forward(
                 1 + ExpFunctor<PredT>::Forward(prediction - 2 * prediction * (prediction >= 0)));
  }
};

template<typename PredT, typename LabelT>
struct SigmoidCrossEntropyGradFunctor {
  OF_DEVICE_FUNC PredT operator()(const PredT prediction, const LabelT label,
                                  const PredT loss_diff) const {
    return loss_diff * (1.f / (1.f + ExpFunctor<PredT>::Forward(-prediction)) - label);
  }
};

namespace {
template<DeviceType device_type, template<typename, typename> class Opt, typename PredT,
         typename LabelT>
struct ElemwiseSigmoidCrossEntropyGradFunctor final {
  void operator()(DeviceCtx* ctx, int64_t n, PredT* prediction_diff, const PredT* prediction,
                  const LabelT* label, const PredT* loss_diff);
};

template<DeviceType device_type, template<typename, typename> class Opt, typename PredT,
         typename LabelT>
struct ElemwiseSigmoidCrossEntropyFunctor final {
  void operator()(DeviceCtx* ctx, int64_t n, PredT* loss, const PredT* prediction,
                  const LabelT* label);
};
}  // namespace

template<DeviceType device_type, template<typename, typename> class Opt, typename PredT,
         typename LabelT>
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
    ElemwiseSigmoidCrossEntropyFunctor<device_type, Opt, PredT, LabelT>()(
        ctx->device_ctx(), n, loss->mut_dptr<PredT>(), prediction->dptr<PredT>(),
        label->dptr<LabelT>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(device_type, dtype, ltype)                      \
  REGISTER_USER_KERNEL("sigmoid_cross_entropy")                                               \
      .SetCreateFn<                                                                           \
          SigmoidCrossEntropyKernel<device_type, SigmoidCrossEntropyFunctor, dtype, ltype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device_type)                               \
                       & (user_op::HobDataType("label", 0) == GetDataType<ltype>::value)      \
                       & (user_op::HobDataType("loss", 0) == GetDataType<dtype>::value));

template<DeviceType device_type, template<typename, typename> class Opt, typename PredT,
         typename LabelT>
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
    ElemwiseSigmoidCrossEntropyGradFunctor<device_type, Opt, PredT, LabelT>()(
        ctx->device_ctx(), n, prediction_diff->mut_dptr<PredT>(), prediction->dptr<PredT>(),
        label->dptr<LabelT>(), loss_diff->dptr<PredT>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(device_type, dtype, ltype)                 \
  REGISTER_USER_KERNEL("sigmoid_cross_entropy_grad")                                          \
      .SetCreateFn<SigmoidCrossEntropyGradKernel<device_type, SigmoidCrossEntropyGradFunctor, \
                                                 dtype, ltype>>()                             \
      .SetIsMatchedHob(                                                                       \
          (user_op::HobDeviceTag() == device_type)                                            \
          & (user_op::HobDataType("label", 0) == GetDataType<ltype>::value)                   \
          & (user_op::HobDataType("prediction_diff", 0) == GetDataType<dtype>::value));

}  // namespace oneflow
#endif  // ONEFLOW_USER_KERNELS_SIGMOID_CROSS_ENTROPY_KERNEL_H_
