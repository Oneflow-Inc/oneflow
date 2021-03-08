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
Maybe<void> InferForwardTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  const int32_t dimension = ctx->Attr<int32_t>("dimension");
  const ShapeView& in_shape = in->shape();
  const int32_t in_dim = in_shape.NumAxes();
  DimVector out_dim_vec = {0};

  if (in_dim == 1) {
    int32_t out_tensor_size = in_shape.At(0) + std::abs(dimension);
    out_dim_vec[0] = out_tensor_size;
    out_dim_vec.push_back(out_tensor_size);
  } else {
    if (dimension >= 0) {
      out_dim_vec[0] = std::min(in_shape.At(0), in_shape.At(1) - dimension);
    } else {
      out_dim_vec[0] = std::min(in_shape.At(0) + dimension, in_shape.At(1));
    }
  }

  user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  out_desc->set_is_dynamic(false);
  *out_desc->mut_shape() = Shape(out_dim_vec);
  *out_desc->mut_data_type() = oneflow::kFloat;
  *out_desc->mut_data_type() = in->data_type();
  return Maybe<void>::Ok();
}

Maybe<void> InferBackwardTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  const user_op::TensorDesc* dy_desc = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
  const Shape& in_shape = in->shape();
  user_op::TensorDesc* dx_desc = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
  *dx_desc->mut_shape() = Shape(in_shape.dim_vec());
  *dx_desc->mut_data_type() = dy_desc->data_type();
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("diag")
    .Input("in")
    .Output("out")
    .Attr<int32_t>("dimension", 0)
    .SetTensorDescInferFn(InferForwardTensorDesc)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      int32_t axis = in_tensor.shape().NumAxes();
      FOR_RANGE(int32_t, i, 0, axis) {
        if (i == axis) { continue; }
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("diag_grad")
    .Input("dy")
    .Input("in")
    .Attr<int32_t>("dimension", 0)
    .Output("dx")
    .SetTensorDescInferFn(InferBackwardTensorDesc)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& y_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0);
      int32_t axis = y_tensor.shape().NumAxes();
      FOR_RANGE(int32_t, i, 0, axis) {
        if (i == axis) { continue; }
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("diag").SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
  const auto grad_op_name = ctx->FwOp().op_name() + "_grad";
  ctx->DefineOp(grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("diag_grad")
        .InputBind("in", ctx->FwOp().input("in", 0))
        .InputBind("dy", ctx->FwOp().output_grad("out", 0))
        .Attr<int32_t>("dimension", ctx->FwOp().attr<int32_t>("dimension"))
        .Output("dx")
        .Build();
  });

  ctx->FwOp().InputGradBind(user_op::OpArg("in", 0), [&ctx, &grad_op_name]() -> const std::string& {
    return ctx->GetOp(grad_op_name).output("dx", 0);
  });
});
}  // namespace oneflow
