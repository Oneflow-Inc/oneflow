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

/* static */ Maybe<void> FusedApplyRotaryEmbOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& cos_desc = ctx->InputTensorDesc("cos", 0);
  const user_op::TensorDesc& sin_desc = ctx->InputTensorDesc("sin", 0);
  const std::string& layout = ctx->Attr<std::string>("layout");

  CHECK_EQ_OR_RETURN(cos_desc.shape().NumAxes(), 2);
  CHECK_EQ_OR_RETURN(sin_desc.shape().NumAxes(), 2);

  CHECK_OR_RETURN(cos_desc.shape() == sin_desc.shape());

  if (x_desc.shape().NumAxes() == 3) {
    if (layout == "BM(HK)") {
      CHECK_EQ_OR_RETURN(cos_desc.shape().At(0), x_desc.shape().At(1));
      CHECK_EQ_OR_RETURN(x_desc.shape().At(2) % cos_desc.shape().At(1), 0);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Not Supported layout of 3-dim x, could be BM(HK)";
    }
  } else if (x_desc.shape().NumAxes() == 4) {
    if (layout == "BHMK") {
      CHECK_EQ_OR_RETURN(cos_desc.shape().At(0), x_desc.shape().At(2));
      CHECK_EQ_OR_RETURN(cos_desc.shape().At(1), x_desc.shape().At(3));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Not Supported layout of 4-dim x, could be BHMK";
    }
  } else {
    UNIMPLEMENTED_THEN_RETURN() << "Not Supported num_dims of x, should be 3 or 4.";
  }

  ctx->SetOutputShape("out", 0, x_desc.shape());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedApplyRotaryEmbOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedApplyRotaryEmbOp::GetSbp(user_op::SbpContext* ctx) {
  // (b, h, s, d) elemntwise-multiply (s, d)

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedApplyRotaryEmbOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& first_in_desc = ctx->InputTensorDesc("x", 0);

  for (const auto& in_arg_pair : ctx->inputs()) {
    const user_op::TensorDesc& in_desc =
        ctx->InputTensorDesc(in_arg_pair.first, in_arg_pair.second);
    CHECK_EQ_OR_RETURN(in_desc.data_type(), first_in_desc.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(first_in_desc.data_type())
        << ", but got " << DataType_Name(in_desc.data_type());
  }

  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_data_type(first_in_desc.data_type());

  return Maybe<void>::Ok();
}

}  // namespace oneflow
