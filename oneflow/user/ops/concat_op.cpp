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
  const user_op::TensorDesc* first_in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  const int64_t axis = ctx->Attr<int64_t>("axis");
  CHECK_GE_OR_RETURN(axis, 0);
  CHECK_LT_OR_RETURN(axis, first_in_desc->shape().NumAxes());
  DimVector out_dim_vec = first_in_desc->shape().dim_vec();
  out_dim_vec.at(axis) = 0;
  int64_t dynamic_dim_size = 0;
  for (const auto& in_arg_pair : ctx->inputs()) {
    const user_op::TensorDesc* in_desc =
        ctx->TensorDesc4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
    CHECK_EQ_OR_RETURN(in_desc->data_type(), first_in_desc->data_type());
    CHECK_EQ_OR_RETURN(in_desc->shape().NumAxes(), first_in_desc->shape().NumAxes());
    FOR_RANGE(int64_t, i, 0, in_desc->shape().NumAxes()) {
      if (i == axis) {
        if (in_desc->is_dynamic()) {
          dynamic_dim_size += in_desc->shape().At(i);
        } else {
          out_dim_vec.at(axis) += in_desc->shape().At(i);
        }
      } else {
        CHECK_EQ_OR_RETURN(in_desc->shape().At(i), out_dim_vec.at(i));
      }
    }
  }

  user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  const int64_t max_dim_size = ctx->Attr<int64_t>("max_dim_size");
  CHECK_LE_OR_RETURN(out_dim_vec.at(axis), max_dim_size);
  if (dynamic_dim_size == 0) {
    out_desc->set_is_dynamic(false);
  } else {
    out_desc->set_is_dynamic(true);
    out_dim_vec.at(axis) = max_dim_size;
  }
  *out_desc->mut_shape() = Shape(out_dim_vec);
  *out_desc->mut_data_type() = first_in_desc->data_type();
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignature(user_op::SbpContext* ctx) {
  const int64_t axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc& first_in_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, first_in_desc.shape().NumAxes()) {
    if (i == axis) { continue; }
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

void GenGrapOp(const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
  bool need_grad = false;
  const int32_t in_size = op.input_size("in");
  FOR_RANGE(int32_t, i, 0, in_size) {
    if (op.NeedGenGradTensor4OpInput("in", i)) { need_grad = true; }
  }
  if (need_grad) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    builder = builder.Op("split_like");
    FOR_RANGE(int32_t, i, 0, in_size) { builder = builder.Input("like", op.input("in", i)); }
    user_op::UserOpConfWrapper grad_op = builder.Input("in", op.GetGradTensorWithOpOutput("out", 0))
                                             .Output("out", in_size)
                                             .Attr("axis", op.attr<int64_t>("axis"))
                                             .Build();

    FOR_RANGE(int32_t, i, 0, in_size) {
      if (op.NeedGenGradTensor4OpInput("in", i)) {
        op.BindGradTensorWithOpInput(grad_op.output("out", i), "in", i);
      }
    }
    AddOp(grad_op);
  }
}

}  // namespace

REGISTER_USER_OP("concat")
    .InputWithMinimum("in", 2)
    .Output("out")
    .Attr<int64_t>("axis")
    .Attr<int64_t>("max_dim_size")
    .SetTensorDescInferFn(InferTensorDesc)
    .SetGetSbpFn(GetSbpSignature);

REGISTER_USER_OP_GRAD("concat").SetGenBackwardOpConfFn(GenGrapOp);

}  // namespace oneflow
