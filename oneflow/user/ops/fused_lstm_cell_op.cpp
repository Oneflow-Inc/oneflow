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
#include "oneflow/user/kernels/expand_kernel_utils.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> FusedLstmCellOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& cx_shape = ctx->InputShape("cx", 0);
  *ctx->OutputShape("hy", 0) = cx_shape;
  *ctx->OutputShape("cy", 0) = cx_shape;
  *ctx->OutputShape("workspace", 0) = ctx->InputShape("input_gates", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedLstmCellOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedLstmCellOp::GetSbp(user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedLstmCellOp::InferDataType(user_op::InferContext* ctx) {
  const oneflow::DataType& in_types = ctx->InputDType("cx", 0);
  *ctx->OutputDType("hy", 0) = in_types;
  *ctx->OutputDType("cy", 0) = in_types;
  *ctx->OutputDType("workspace", 0) = in_types;
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedLstmCellGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("grad_gates", 0) = ctx->InputShape("workspace", 0);

  if (ctx->has_output("grad_cx", 0)) { *ctx->OutputShape("grad_cx", 0) = ctx->InputShape("cx", 0); }

  if (ctx->has_output("grad_bias", 0)) {
    std::vector<int32_t> bias_shape;
    bias_shape.push_back(ctx->InputShape("workspace", 0).At(1));
    DimVector dim_vec(bias_shape.begin(), bias_shape.end());
    *ctx->OutputShape("grad_bias", 0) = Shape(dim_vec);
  }

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedLstmCellGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedLstmCellGradOp::GetSbp(user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedLstmCellGradOp::InferDataType(user_op::InferContext* ctx) {
  const oneflow::DataType& in_types = ctx->InputDType("grad_hy", 0);
  *ctx->OutputDType("grad_gates", 0) = in_types;
  if (ctx->has_output("grad_cx", 0)) { *ctx->OutputDType("grad_cx", 0) = in_types; }
  if (ctx->has_output("grad_bias", 0)) { *ctx->OutputDType("grad_bias", 0) = in_types; }
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("fused_lstm_cell")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) -> Maybe<void> {
      const auto grad_op_name = ctx->FwOp().op_name() + "_grad";
      ctx->DefineOp(grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        builder.OpTypeName("fused_lstm_cell_grad")
            .InputBind("grad_hy", ctx->FwOp().output_grad("hy", 0))
            .InputBind("grad_cy", ctx->FwOp().output_grad("cy", 0))
            .InputBind("cx", ctx->FwOp().input("cx", 0))
            .InputBind("cy", ctx->FwOp().output("cy", 0))
            .InputBind("workspace", ctx->FwOp().output("workspace", 0))
            .Output("grad_gates");

        if (ctx->FwOp().NeedGenGradTensor4OpInput("cx", 0)) { builder.Output("grad_cx"); }

        if (ctx->FwOp().user_op_conf().has_input("input_bias", 0)
            && ctx->FwOp().user_op_conf().has_input("hidden_bias", 0)) {
          builder.Output("grad_bias");
        }
        return builder.Build();
      });

      ctx->FwOp().InputGradBind(user_op::OpArg("input_gates", 0),
                                [&ctx, &grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(grad_op_name).output("grad_gates", 0);
                                });
      ctx->FwOp().InputGradBind(user_op::OpArg("hidden_gates", 0),
                                [&ctx, &grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(grad_op_name).output("grad_gates", 0);
                                });

      if (ctx->FwOp().NeedGenGradTensor4OpInput("cx", 0)) {
        ctx->FwOp().InputGradBind(user_op::OpArg("cx", 0),
                                  [&ctx, &grad_op_name]() -> const std::string& {
                                    return ctx->GetOp(grad_op_name).output("grad_cx", 0);
                                  });
      }

      if (ctx->FwOp().user_op_conf().has_input("input_bias", 0)) {
        ctx->FwOp().InputGradBind(user_op::OpArg("input_bias", 0),
                                  [&ctx, &grad_op_name]() -> const std::string& {
                                    return ctx->GetOp(grad_op_name).output("grad_bias", 0);
                                  });
        ctx->FwOp().InputGradBind(user_op::OpArg("hidden_bias", 0),
                                  [&ctx, &grad_op_name]() -> const std::string& {
                                    return ctx->GetOp(grad_op_name).output("grad_bias", 0);
                                  });
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
