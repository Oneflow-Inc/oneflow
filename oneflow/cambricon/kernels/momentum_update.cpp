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
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"

namespace oneflow {

class MluMomentumUpdateKernel final : public user_op::OpKernel {
  void Compute(user_op::KernelComputeContext* ctx) const override {
    float learning_rate_val = ctx->Attr<float>("learning_rate_val");
    double scale = ctx->Attr<double>("scale");
    float l1 = ctx->Attr<float>("l1");
    float l2 = ctx->Attr<float>("l2");
    float beta = ctx->Attr<float>("beta");
    const float dampening = ctx->Attr<float>("dampening");
    const bool nesterov = ctx->Attr<bool>("nesterov");
    const bool maximize = ctx->Attr<bool>("maximize");
    float weight_decay = ctx->Attr<float>("weight_decay");
    const auto lr_scale = ctx->Attr<float>("learning_rate_scale");
    CHECK_EQ(scale, 1);
    CHECK_EQ(l1, 0);
    CHECK_EQ(l2, 0);
    CHECK_EQ(dampening, 0);
    CHECK(!maximize);
    CHECK_EQ(weight_decay, 0);
    CHECK_EQ(lr_scale, 1);

    const user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    user_op::Tensor* momentum = ctx->Tensor4ArgNameAndIndex("momentum", 0);
    CHECK(!ctx->has_input("learning_rate", 0));
    CHECK(!ctx->has_input("scale_by_tensor", 0));
    CHECK(!ctx->has_input("skip_if", 0));

    CnnlTensorDescriptor model_desc(model), accum_desc(momentum), diff_desc(model_diff);

    auto stream = ctx->stream()->As<ep::MluStream>();
    std::unique_ptr<ep::primitive::Fill> primitive_fill =
        ep::primitive::NewPrimitive<ep::primitive::FillFactory>(DeviceType::kMLU, DataType::kFloat);
    CnnlWorkspace lr_mlu(stream, sizeof(float));
    primitive_fill->Launch(stream, lr_mlu.dptr(), learning_rate_val, 1);
    CnnlWorkspace beta_mlu(stream, sizeof(float));
    primitive_fill->Launch(stream, beta_mlu.dptr(), beta, 1);

    OF_CNNL_CHECK(cnnlKerasMomentum(stream->cnnl_handle(), model_desc.desc(), model->mut_raw_dptr(),
                                    accum_desc.desc(), momentum->mut_raw_dptr(), diff_desc.desc(),
                                    model_diff->raw_dptr(), lr_mlu.dptr(), beta_mlu.dptr(),
                                    nesterov));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("momentum_update")
    .SetCreateFn<MluMomentumUpdateKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU));

}  // namespace oneflow
