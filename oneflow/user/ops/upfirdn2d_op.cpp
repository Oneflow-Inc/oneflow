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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> CalcUpFirDnOut(int64_t input_size, int32_t up, int32_t down, int32_t pad_h,
                           int32_t pad_w, int32_t kernel_size, int64_t* output_size) {
  if (output_size) {
    *output_size = (input_size * up + pad_h + pad_w - kernel_size) / down + 1;
    CHECK_GE_OR_RETURN((*output_size), 0);
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> Upfirdn2dOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input = ctx->InputTensorDesc("input", 0);
  const Shape& input_shape = ctx->InputShape("input", 0);
  int32_t input_ndim = input_shape.NumAxes();
  CHECK_EQ_OR_RETURN(4, input_ndim)
      << "UpFirDn2D op's input shape ndim should equal to " << 4
      << " ,but got: " << input_ndim;

  const Shape& kernel_shape = ctx->InputShape("kernel", 0);
  CHECK_EQ_OR_RETURN(2, kernel_shape.size());

  auto up = ctx->Attr<std::vector<int32_t>>("up");
  auto down = ctx->Attr<std::vector<int32_t>>("down");
  auto pad = ctx->Attr<std::vector<int32_t>>("pad");

  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  DimVector out_shape(4);
  out_shape.at(0) = input_shape.At(0) * input_shape.At(1);
  out_shape.at(3) = 1;
  for (int32_t i = 0; i < 2; ++i) {
    JUST(CalcUpFirDnOut(input_shape.At(i + 2), up[i], down[i], pad[i * 2], pad[i * 2 + 1],
                        kernel_shape.At(i), &out_shape.at(i + 1)));
  }

  out->set_is_dynamic(input.is_dynamic());
  out->set_shape(Shape(out_shape));

  return Maybe<void>::Ok();
}

Maybe<void> Upfirdn2dOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return Upfirdn2dOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> Upfirdn2dOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& kernel = ctx->InputTensorDesc("kernel", 0);

  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);

  CHECK_EQ_OR_RETURN(input.data_type(), kernel.data_type());
  out->set_data_type(input.data_type());
  return Maybe<void>::Ok();
}

Maybe<void> Upfirdn2dOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("input", 0), 0)
      .Broadcast(user_op::OpArg("kernel", 0))
      .Split(user_op::OpArg("out", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
