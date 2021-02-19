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
#ifndef _ONEFLOW_USER_KERNELS_SIGMOID_CROSS_ENTROPY_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_SIGMOID_CROSS_ENTROPY_KERNEL_H_
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

template<typename T>
struct SigmoidCrossEntropyFunctor {
  OF_DEVICE_FUNC T operator()(const T prediction, const T label) const {
    return -1.f * prediction * (label - (prediction >= 0))
           + logf(1 + expf(prediction - 2 * prediction * (prediction >= 0)));
  }
};

template<typename T>
struct SigmoidCrossEntropyGradFunctor {
  OF_DEVICE_FUNC T operator()(const T prediction, const T label) const {
    return 1.f / (1.f + expf(-prediction)) - label;
  }
};

namespace {
template<DeviceType device_type, template<typename> class Opt, typename T>
struct ElemwiseSigmoidCrossEntropyGradFunctor final {
  void operator()(DeviceCtx* ctx, int64_t n, T* prediction_diff, const T* prediction,
                  const T* label);
};

template<DeviceType device_type, template<typename> class Opt, typename T>
struct ElemwiseSigmoidCrossEntropyFunctor final {
  void operator()(DeviceCtx* ctx, int64_t n, T* loss, const T* prediction, const T* label);
};
}  // namespace

template<DeviceType device_type, template<typename> class Opt, typename T>
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
    ElemwiseSigmoidCrossEntropyFunctor<device_type, Opt, T>()(
        ctx->device_ctx(), n, loss->mut_dptr<T>(), prediction->dptr<T>(), label->dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SIGMOID_CROSS_ENTROPY_KERNEL(device_type, dtype)                               \
  REGISTER_USER_KERNEL("sigmoid_cross_entropy")                                                 \
      .SetCreateFn<SigmoidCrossEntropyKernel<device_type, SigmoidCrossEntropyFunctor, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device_type)                                 \
                       & (user_op::HobDataType("label", 0) == GetDataType<dtype>::value)        \
                       & (user_op::HobDataType("loss", 0) == GetDataType<dtype>::value));

template<DeviceType device_type, template<typename> class Opt, typename T>
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
    ElemwiseSigmoidCrossEntropyGradFunctor<device_type, Opt, T>()(
        ctx->device_ctx(), n, prediction_diff->mut_dptr<T>(), prediction->dptr<T>(),
        label->dptr<T>());
    KernelUtil<device_type, T>::Mul(ctx->device_ctx(), n, prediction_diff->dptr<T>(),
                                    loss_diff->dptr<T>(), prediction_diff->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SIGMOID_CROSS_ENTROPY_GRAD_KERNEL(device_type, dtype)                         \
  REGISTER_USER_KERNEL("sigmoid_cross_entropy_grad")                                           \
      .SetCreateFn<                                                                            \
          SigmoidCrossEntropyGradKernel<device_type, SigmoidCrossEntropyGradFunctor, dtype>>() \
      .SetIsMatchedHob(                                                                        \
          (user_op::HobDeviceTag() == device_type)                                             \
          & (user_op::HobDataType("label", 0) == GetDataType<dtype>::value)                    \
          & (user_op::HobDataType("prediction_diff", 0) == GetDataType<dtype>::value));

}  // namespace oneflow
#endif  // _ONEFLOW_USER_KERNELS_SIGMOID_CROSS_ENTROPY_KERNEL_H_
