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
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

class MluActivationKernel final : public user_op::OpKernel {
 public:
  MluActivationKernel() = default;
  ~MluActivationKernel() = default;

  using ComputeImpl = std::function<void(user_op::KernelComputeContext* ctx)>;
  explicit MluActivationKernel(ComputeImpl compute_impl) : compute_impl_(compute_impl) {}

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override { compute_impl_(ctx); }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  ComputeImpl compute_impl_;
};

void CnnlActivationForward(user_op::KernelComputeContext* ctx, const user_op::Tensor* in,
                           user_op::Tensor* out, const CnnlActivationDescriptor& activation_desc) {
  CnnlTensorDescriptor input_desc, output_desc;
  input_desc.set(in);
  output_desc.set(out);
  OF_CNNL_CHECK(cnnlActivationForward(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                      activation_desc.desc(), nullptr, input_desc.desc(),
                                      in->dptr(), nullptr, output_desc.desc(), out->mut_dptr()));
}

inline auto BaseActivationIsMatched(const std::string& input_name) {
  return (user_op::HobDeviceType() == DeviceType::kMLU)
         && ((user_op::HobDataType(input_name, 0) == kFloat)
             || (user_op::HobDataType(input_name, 0) == kFloat16));
}

REGISTER_USER_KERNEL("relu")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<MluActivationKernel>([](user_op::KernelComputeContext* ctx) {
        const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("x", 0);
        user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("y", 0);
        CnnlActivationDescriptor activation_desc;
        activation_desc.set(CNNL_ACTIVATION_RELU, /*prefer=*/CNNL_ACTIVATION_HIGH_PRECISION,
                            /*nanProp=*/CNNL_NOT_PROPAGATE_NAN, /*ceof=*/1.0);
        CnnlActivationForward(ctx, in, out, activation_desc);
      });
    })
    .SetIsMatchedHob(BaseActivationIsMatched("x"));

REGISTER_USER_KERNEL("gelu")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<MluActivationKernel>([](user_op::KernelComputeContext* ctx) {
        const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
        user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
        CnnlActivationDescriptor activation_desc;
        activation_desc.set(CNNL_ACTIVATION_GELU, /*prefer=*/CNNL_ACTIVATION_HIGH_PRECISION,
                            /*nanProp=*/CNNL_NOT_PROPAGATE_NAN, /*ceof=*/1.0);
        CnnlActivationForward(ctx, in, out, activation_desc);
      });
    })
    .SetIsMatchedHob(BaseActivationIsMatched("in"));

}  // namespace oneflow
