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

/* static */ Maybe<void> TrilFlatOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  int64_t input_num_axes = in.shape().NumAxes();
  CHECK_GE_OR_RETURN(input_num_axes, 2);

  const int64_t diagonal = ctx->Attr<int64_t>("diagonal");

  DimVector out_dim_vec = {};
  FOR_RANGE(int64_t, i, 0, input_num_axes - 2 ) { out_dim_vec.push_back(in.shape().At(i)); }
  
  const int64_t m = in.shape().At(input_num_axes - 2);
  const int64_t n = in.shape().At(input_num_axes - 1);
  int64_t last_dim = 0;
  FOR_RANGE(int64_t, row, 0, m) {
    FOR_RANGE(int64_t, col, 0, n) {
      if(row - col >= diagonal) {
        ++last_dim;
      }
    }
  }
  out_dim_vec.push_back(last_dim);

  user_op::TensorDesc* out_desc = ctx->OutputTensorDesc("out", 0);
  out_desc->set_is_dynamic(false);
  *out_desc->mut_shape() = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TrilFlatOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> TrilFlatOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in.shape().NumAxes() - 2) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> TrilFlatOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> TrilFlatGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  const Shape& in_shape = x.shape();
  user_op::TensorDesc* dx = ctx->OutputTensorDesc("dx", 0);
  *dx->mut_shape() = Shape(in_shape.dim_vec());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TrilFlatGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> TrilFlatGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& dy = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0);
  FOR_RANGE(int64_t, i, 0, dy.shape().NumAxes() - 1) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }  
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> TrilFlatGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("tril_flat")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) -> Maybe<void> {
      const auto grad_op_name = ctx->FwOp().op_name() + "_grad";
      ctx->DefineOp(grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("tril_flat_grad")
            .InputBind("x", ctx->FwOp().input("in", 0))
            .InputBind("dy", ctx->FwOp().output_grad("out", 0))
            .Attr<int32_t>("diagonal", ctx->FwOp().attr<int32_t>("diagonal"))
            .Output("dx")
            .Build();
      });

      ctx->FwOp().InputGradBind(user_op::OpArg("in", 0),
                                [&ctx, &grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(grad_op_name).output("dx", 0);
                                });
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
