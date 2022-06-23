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

/* static */ Maybe<void> AddNOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto& in_0 = ctx->InputTensorDesc("in", 0);
  auto* out = ctx->OutputTensorDesc("out", 0);
  CHECK_NOTNULL_OR_RETURN(out);
  for (const auto& pair : ctx->inputs()) {
    const auto& cur_in = ctx->InputTensorDesc(pair.first, pair.second);
    if (in_0.shape().NumAxes() > 0 && cur_in.shape().NumAxes() > 0) {
      CHECK_EQ_OR_RETURN(in_0.shape(), cur_in.shape());
    }
  }
  *out->mut_shape() = in_0.shape();
  *out->mut_is_dynamic() = in_0.is_dynamic();
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> AddNOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> AddNOp::GetSbp(user_op::SbpContext* ctx) {
  int64_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
  for (int64_t i = 0; i < num_axes; ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(user_op::OpArg("out", 0), i).Build();
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(user_op::OpArg("out", 0)).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AddNOp::InferDataType(user_op::InferContext* ctx) {
  const auto& in_0 = ctx->InputTensorDesc("in", 0);
  auto* out = ctx->OutputTensorDesc("out", 0);
  CHECK_NOTNULL_OR_RETURN(out);
  for (const auto& pair : ctx->inputs()) {
    const auto& cur_in = ctx->InputTensorDesc(pair.first, pair.second);
    CHECK_EQ_OR_RETURN(in_0.data_type(), cur_in.data_type()) << ctx->op_name();
  }
  *out->mut_data_type() = in_0.data_type();
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> AddNOp::CheckAttr(const user_op::UserOpDefWrapper&,
                                         const user_op::UserOpConfWrapper& op_conf) {
  CHECK_OR_RETURN(op_conf.input_size("in") >= 2);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("add_n").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                         user_op::AddOpFn AddOp) -> Maybe<void> {
  int32_t in_size = op.input_size("in");
  for (int i = 0; i < in_size; ++i) {
    if (op.NeedGenGradTensor4OpInput("in", i)) {
      op.BindGradTensorWithOpInput(op.GetGradTensorWithOpOutput("out", 0), "in", i);
    }
  }
  return Maybe<void>::Ok();
});

}  // namespace oneflow
