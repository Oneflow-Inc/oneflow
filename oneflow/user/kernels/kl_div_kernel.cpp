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
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/user/kernels/loss_kernel_util.h"

namespace oneflow {
namespace user_op {
namespace {

using namespace loss;

template<typename T>
void ComputeKLDivOut(int64_t elem_cnt, const T* input, const T* target, T* out,
                     const bool log_target) {
  if (log_target) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out[i] = std::exp(target[i]) * (target[i] - input[i]); }
  } else {
    FOR_RANGE(int64_t, i, 0, elem_cnt) {
      const auto out_val = target[i] * (SafeLog(target[i]) - input[i]);
      out[i] = target[i] > 0 ? out_val : static_cast<T>(0);
    }
  }
}
template<typename T>
void ComputeKLDivGradOut(int64_t elem_cnt, const T* input, const T* target, const T* dy, T* dx,
                         const bool log_target, const ReductionType reduction_type) {
#define SET_DY_VAL const T dy_val = reduction_type == ReductionType::kNone ? dy[i] : *dy;
#define DEAL_REDUCE_MEAN \
  if (reduction_type == ReductionType::kMean) { dx[i] /= elem_cnt; };

  {
    if (log_target) {
      FOR_RANGE(int64_t, i, 0, elem_cnt) {
        SET_DY_VAL
        dx[i] = -std::exp(target[i]) * dy_val;
        DEAL_REDUCE_MEAN
      }
    } else {
      FOR_RANGE(int64_t, i, 0, elem_cnt) {
        SET_DY_VAL
        dx[i] = target[i] > 0 ? -target[i] * dy_val : 0;
        DEAL_REDUCE_MEAN
      }
    }
  }

#undef SET_DY_VAL
#undef DEAL_REDUCE_MEAN
}

template<typename T>
class KLDivKernel final : public user_op::OpKernel {
 public:
  KLDivKernel() = default;
  ~KLDivKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto* tmp_buffer_blob = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));
    const bool log_target = ctx->Attr<bool>("log_target");

    const int64_t elem_cnt = input_blob->shape().elem_cnt();

    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* out = out_blob->mut_dptr<T>();
    T* tmp_buffer = tmp_buffer_blob->mut_dptr<T>();
    T* tmp_out = tmp_buffer;

    ComputeKLDivOut(elem_cnt, input, target, reduction == ReductionType::kNone ? out : tmp_out,
                    log_target);

    ApplyLossReductionIfNeed<T>(elem_cnt, tmp_out, out, reduction);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class KLDivGradKernel final : public user_op::OpKernel {
 public:
  KLDivGradKernel() = default;
  ~KLDivGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    const auto* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));
    const bool log_target = ctx->Attr<bool>("log_target");

    const int64_t elem_cnt = input_blob->shape().elem_cnt();

    const T* dy = dy_blob->dptr<T>();
    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* dx = dx_blob->mut_dptr<T>();
    ComputeKLDivGradOut(elem_cnt, input, target, dy, dx, log_target, reduction);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_KL_DIV_KERNEL(dtype)                                                     \
  REGISTER_USER_KERNEL("kl_div")                                                          \
      .SetCreateFn<KLDivKernel<dtype>>()                                                  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU)                      \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       & (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))   \
      .SetInferTmpSizeFn(loss::GenDefaultInferTmpSizeFn<dtype>());

#define REGISTER_KL_DIV_GRAD_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("kl_div_grad")                                                     \
      .SetCreateFn<KLDivGradKernel<dtype>>()                                              \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU)                      \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       & (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)     \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_KL_DIV_KERNEL(float)
REGISTER_KL_DIV_KERNEL(double)
REGISTER_KL_DIV_GRAD_KERNEL(float)
REGISTER_KL_DIV_GRAD_KERNEL(double)

}  // namespace user_op
}  // namespace oneflow
