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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/loss_kernel_util.h"

namespace oneflow {
namespace user_op {

namespace {

using namespace loss;

template<typename T>
void ComputeSmoothL1Out(int64_t elem_cnt, const T* prediction, const T* label, T* out,
                        const float beta) {
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    const T abs_diff = std::abs(prediction[i] - label[i]);
    if (abs_diff < beta) {
      out[i] = 0.5 * abs_diff * abs_diff / beta;
    } else {
      out[i] = abs_diff - 0.5 * beta;
    }
  }
}
template<typename T>
void ComputeSmoothL1GradOut(int64_t elem_cnt, const T* prediction, const T* label,
                            const T* loss_grad, T* prediction_grad, const float beta,
                            const ReductionType reduction_type) {
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    const T diff = prediction[i] - label[i];
    const T abs_diff = std::abs(diff);
    if (abs_diff < beta) {
      prediction_grad[i] = diff / beta;
    } else {
      prediction_grad[i] = (diff > GetZeroVal<T>()) - (diff < GetZeroVal<T>());
    }
    const T loss_grad_val = reduction_type == ReductionType::kNone ? loss_grad[i] : *loss_grad;
    prediction_grad[i] = prediction_grad[i] * loss_grad_val;
    if (reduction_type == ReductionType::kMean) { prediction_grad[i] /= elem_cnt; };
  }
}
template<typename T>
class SmoothL1LossKernel final : public user_op::OpKernel {
 public:
  SmoothL1LossKernel() = default;
  ~SmoothL1LossKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* prediction_blob = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const auto* label_blob = ctx->Tensor4ArgNameAndIndex("label", 0);
    auto* loss_blob = ctx->Tensor4ArgNameAndIndex("loss", 0);
    auto* tmp_buffer_blob = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));
    const float beta = ctx->Attr<float>("beta");

    const int64_t elem_cnt = prediction_blob->shape().elem_cnt();

    const T* prediction = prediction_blob->dptr<T>();
    const T* label = label_blob->dptr<T>();
    T* loss = loss_blob->mut_dptr<T>();
    T* tmp_buffer = tmp_buffer_blob->mut_dptr<T>();
    T* tmp_out = tmp_buffer;

    ComputeSmoothL1Out(elem_cnt, prediction, label,
                       reduction == ReductionType::kNone ? loss : tmp_out, beta);
    ApplyLossReductionIfNeed(elem_cnt, tmp_out, loss, reduction);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class SmoothL1LossGradKernel final : public user_op::OpKernel {
 public:
  SmoothL1LossGradKernel() = default;
  ~SmoothL1LossGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* prediction_blob = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const auto* label_blob = ctx->Tensor4ArgNameAndIndex("label", 0);
    const auto* loss_grad_blob = ctx->Tensor4ArgNameAndIndex("loss_grad", 0);
    auto* prediction_grad_blob = ctx->Tensor4ArgNameAndIndex("prediction_grad", 0);
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));
    const float beta = ctx->Attr<float>("beta");

    const int64_t elem_cnt = prediction_blob->shape().elem_cnt();

    const T* loss_grad = loss_grad_blob->dptr<T>();
    const T* prediction = prediction_blob->dptr<T>();
    const T* label = label_blob->dptr<T>();
    T* prediction_grad = prediction_grad_blob->mut_dptr<T>();

    ComputeSmoothL1GradOut(elem_cnt, prediction, label, loss_grad, prediction_grad, beta,
                           reduction);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_SMOOTH_L1_LOSS_KERNEL(dtype)                                            \
  REGISTER_USER_KERNEL("smooth_l1_loss")                                                 \
      .SetCreateFn<SmoothL1LossKernel<dtype>>()                                          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                                \
                       & (user_op::HobDataType("loss", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(loss::GenDefaultInferTmpSizeFn<dtype>("prediction"));

#define REGISTER_SMOOTH_L1_LOSS_GRAD_KERNEL(dtype)  \
  REGISTER_USER_KERNEL("smooth_l1_loss_grad")       \
      .SetCreateFn<SmoothL1LossGradKernel<dtype>>() \
      .SetIsMatchedHob(                             \
          (user_op::HobDeviceTag() == "cpu")        \
          & (user_op::HobDataType("prediction_grad", 0) == GetDataType<dtype>::value));

REGISTER_SMOOTH_L1_LOSS_KERNEL(float)
REGISTER_SMOOTH_L1_LOSS_KERNEL(double)
REGISTER_SMOOTH_L1_LOSS_GRAD_KERNEL(float)
REGISTER_SMOOTH_L1_LOSS_GRAD_KERNEL(double)

}  // namespace user_op
}  // namespace oneflow
