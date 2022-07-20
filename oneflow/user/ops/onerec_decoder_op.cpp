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

/* static */ Maybe<void> OnerecDecoderOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
  CHECK_OR_RETURN(in_tensor.shape().NumAxes() == 1 && in_tensor.shape().At(0) >= 1);
  const Shape& static_shape = ctx->Attr<Shape>("static_shape");
  DimVector dim_vec(1 + static_shape.NumAxes());
  dim_vec[0] = in_tensor.shape().At(0);
  FOR_RANGE(int64_t, i, 1, dim_vec.size()) { dim_vec[i] = static_shape.At(i - 1); }
  *out_tensor->mut_shape() = Shape(dim_vec);
  out_tensor->set_is_dynamic(ctx->Attr<bool>("is_dynamic"));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OnerecDecoderOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OnerecDecoderOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("in", 0), 0).Split(user_op::OpArg("out", 0), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OnerecDecoderOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* in_modifier = GetInputArgModifierFn("in", 0);
  CHECK_NOTNULL_OR_RETURN(in_modifier);
  in_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OnerecDecoderOp::ModifyOutputArg(
    const GetOutputArgModifier& GetOutputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  // NOTE(yaochi): refer to tensor_buffer_to_list_of_tensors
  // In order to support global tensor, set set_header_infered_before_compute to false
  // only when is_dynamic == true
  if (conf.attr<bool>("is_dynamic")) {
    FOR_RANGE(int64_t, i, 0, conf.output_size("out")) {
      user_op::OutputArgModifier* out_i_modifier = GetOutputArgModifierFn("out", i);
      CHECK_OR_RETURN(out_i_modifier != nullptr);
      out_i_modifier->set_header_infered_before_compute(false);
    }
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OnerecDecoderOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
  CHECK_OR_RETURN(in_tensor.data_type() == DataType::kTensorBuffer);
  *out_tensor->mut_data_type() = ctx->Attr<DataType>("data_type");
  return Maybe<void>::Ok();
}

}  // namespace oneflow
