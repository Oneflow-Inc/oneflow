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
#include "oneflow/core/framework/device.h"

namespace oneflow {

namespace {

Maybe<const Device> MakeOpDevice(const std::shared_ptr<const Device>& in_device,
                                 const std::shared_ptr<const Device>& out_device) {
  if (in_device->type() == "cuda" && out_device->type() == "cpu") {
    return Device::New("cuda_d2h");
  } else if (in_device->type() == "cpu" && out_device->type() == "cuda") {
    return Device::New("cuda_h2d");
  } else {
    return Device::New(in_device->type());
  }
}

REGISTER_USER_OP("copy")
    .Input("in")
    .Output("out")
    .Attr<std::string>("device_type")
    .Attr<int>("device_id")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Shape4ArgNameAndIndex("out", 0) = *ctx->Shape4ArgNameAndIndex("in", 0);
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn([](user_op::DeviceInferContext* ctx) -> Maybe<const Device> {
      std::shared_ptr<const Device> out_device =
          Device::New(ctx->Attr<std::string>("device_type"), ctx->Attr<int>("device_id"))
              .GetPtrOrThrow();
      *ctx->OutputTensorDevice4ArgNameAndIndex("out", 0) = out_device;
      const std::shared_ptr<const Device>& in_device =
          ctx->InputTensorDevice4ArgNameAndIndex("in", 0);
      return MakeOpDevice(in_device, out_device);
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("copy_grad")
    .Input("out_grad")
    .Input("in")
    .Output("in_grad")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      CHECK_EQ_OR_RETURN(*ctx->Shape4ArgNameAndIndex("in", 0),
                         *ctx->Shape4ArgNameAndIndex("out_grad", 0));
      *ctx->Shape4ArgNameAndIndex("in_grad", 0) = *ctx->Shape4ArgNameAndIndex("out_grad", 0);
      *ctx->IsDynamic4ArgNameAndIndex("in_grad", 0) =
          *ctx->IsDynamic4ArgNameAndIndex("out_grad", 0);
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn([](user_op::DeviceInferContext* ctx) -> Maybe<const Device> {
      *ctx->OutputTensorDevice4ArgNameAndIndex("in_grad", 0) =
          ctx->InputTensorDevice4ArgNameAndIndex("in", 0);
      const auto& in_device = ctx->InputTensorDevice4ArgNameAndIndex("in", 0);
      const auto& out_grad_device = ctx->InputTensorDevice4ArgNameAndIndex("out_grad", 0);
      return MakeOpDevice(out_grad_device, in_device);
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("in", 0),
                         *ctx->Dtype4ArgNameAndIndex("out_grad", 0));
      *ctx->Dtype4ArgNameAndIndex("in_grad", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("copy").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                        user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op =
        builder.Op("copy_grad")
            .Input("out_grad", op.GetGradTensorWithOpOutput("out", 0))
            .Input("in", op.input("in", 0))
            .Output("in_grad")
            .Build();
    AddOp(grad_op);
    op.BindGradTensorWithOpInput(grad_op.output("in_grad", 0), "in", 0);
  }
});
}  // namespace
}  // namespace oneflow
