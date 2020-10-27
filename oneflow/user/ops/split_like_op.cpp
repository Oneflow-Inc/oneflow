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

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const int64_t axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  int64_t dynamic_dim_size = 0;
  int64_t static_dim_size = 0;
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    const user_op::TensorDesc* like_i_desc = ctx->TensorDesc4ArgNameAndIndex("like", i);
    user_op::TensorDesc* out_i_desc = ctx->TensorDesc4ArgNameAndIndex("out", i);
    CHECK_EQ_OR_RETURN(like_i_desc->shape().NumAxes(), in_desc->shape().NumAxes());
    FOR_RANGE(int64_t, j, 0, in_desc->shape().NumAxes()) {
      if (j == axis) {
        if (like_i_desc->is_dynamic()) {
          dynamic_dim_size += like_i_desc->shape().At(j);
        } else {
          static_dim_size += like_i_desc->shape().At(j);
        }
      } else {
        CHECK_EQ_OR_RETURN(in_desc->shape().At(j), like_i_desc->shape().At(j));
      }
    }
    *out_i_desc->mut_shape() = like_i_desc->shape();
    *out_i_desc->mut_data_type() = in_desc->data_type();
    out_i_desc->set_is_dynamic(like_i_desc->is_dynamic());
  }
  if (dynamic_dim_size == 0) {
    CHECK_EQ_OR_RETURN(static_dim_size, in_desc->shape().At(axis));
  } else {
    CHECK_LE_OR_RETURN(static_dim_size, in_desc->shape().At(axis));
  }
  return Maybe<void>::Ok();
}

void SetLikeArgModifier(user_op::GetInputArgModifier GetInputArgModifierFn,
                        const user_op::UserOpConfWrapper& user_op_conf) {
  FOR_RANGE(int32_t, i, 0, user_op_conf.input_size("like")) {
    user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", i);
    CHECK_NOTNULL(like_modifier);
    like_modifier->set_use_header_only(true);
    like_modifier->set_requires_grad(false);
  }
}

Maybe<void> InferBatchAxis(user_op::BatchAxisContext* ctx) {
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    *ctx->BatchAxis4ArgNameAndIndex("out", i) = *ctx->BatchAxis4ArgNameAndIndex("like", i);
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignature(user_op::SbpContext* ctx) {
  const int64_t axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc& in_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_desc.shape().NumAxes()) {
    if (i == axis) { continue; }
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  std::vector<user_op::OpArg> like_arg_vec;
  const size_t like_arg_size = ctx->outputs().size();
  like_arg_vec.reserve(like_arg_size);
  FOR_RANGE(int32_t, i, 0, like_arg_size) { like_arg_vec.push_back(user_op::OpArg("like", i)); }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(like_arg_vec)
      .PartialSum(ctx->outputs())
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .Broadcast(like_arg_vec)
      .PartialSum(ctx->outputs())
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("in", 0))
      .PartialSum(like_arg_vec)
      .Broadcast(ctx->outputs())
      .Build();
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("split_like")
    .Input("in")
    .InputWithMinimum("like", 2)
    .OutputWithMinimum("out", 2)
    .Attr<int64_t>("axis")
    .SetTensorDescInferFn(InferTensorDesc)
    .SetInputArgModifyFn(SetLikeArgModifier)
    .SetBatchAxisInferFn(InferBatchAxis)
    .SetGetSbpFn(GetSbpSignature);

}  // namespace oneflow
