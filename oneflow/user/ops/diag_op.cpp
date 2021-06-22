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

REGISTER_USER_OP("diag")
    .Input("in")
    .Output("out")
    .Attr<int32_t>("diagonal", 0)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      const int32_t diagonal = ctx->Attr<int32_t>("diagonal");
      const ShapeView& in_shape = in->shape();
      const int32_t in_dim = in_shape.NumAxes();
      CHECK_GE_OR_RETURN(in_dim, 1);
      CHECK_LE_OR_RETURN(in_dim, 2);

      DimVector out_dim_vec = {0};
      if (in_dim == 1) {
        int32_t out_tensor_size = in_shape.At(0) + std::abs(diagonal);
        out_dim_vec[0] = out_tensor_size;
        out_dim_vec.push_back(out_tensor_size);
      } else {
        if (diagonal >= 0) {
          out_dim_vec[0] = std::min(in_shape.At(0), in_shape.At(1) - diagonal);
        } else {
          out_dim_vec[0] = std::min(in_shape.At(0) + diagonal, in_shape.At(1));
        }
        CHECK_GT_OR_RETURN(out_dim_vec[0], 0);
      }

      user_op::TensorDesc* out_desc = ctx->OutputTensorDesc("out", 0);
      out_desc->set_is_dynamic(false);
      *out_desc->mut_shape() = Shape(out_dim_vec);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("diag_grad")
    .Input("dy")
    .Input("in")
    .Attr<int32_t>("diagonal", 0)
    .Output("dx")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      const Shape& in_shape = in->shape();
      user_op::TensorDesc* dx_desc = ctx->OutputTensorDesc("dx", 0);
      *dx_desc->mut_shape() = Shape(in_shape.dim_vec());
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("diag").SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
  const auto grad_op_name = ctx->FwOp().op_name() + "_grad";
  ctx->DefineOp(grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("diag_grad")
        .InputBind("in", ctx->FwOp().input("in", 0))
        .InputBind("dy", ctx->FwOp().output_grad("out", 0))
        .Attr<int32_t>("diagonal", ctx->FwOp().attr<int32_t>("diagonal"))
        .Output("dx")
        .Build();
  });

  ctx->FwOp().InputGradBind(user_op::OpArg("in", 0), [&ctx, &grad_op_name]() -> const std::string& {
    return ctx->GetOp(grad_op_name).output("dx", 0);
  });
});
}  // namespace oneflow
