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

/* static */ Maybe<void> FusedGruCellOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& hx_shape = ctx->InputShape("hx", 0);
  *ctx->MutOutputShape("hy", 0) = hx_shape;
  *ctx->MutOutputShape("workspace", 0) = Shape({hx_shape.At(0), hx_shape.At(1) * 5});
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedGruCellOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedGruCellOp::GetSbp(user_op::SbpContext* ctx) {
  // input_gates shape:  [batch_size, hidden_size * 3]
  // hidden_gates shape: [batch_size, hidden_size * 3]
  // hx shape:           [batch_size, hidden_size]
  // input_bias shape:   [hidden_size * 3]
  // hidden_bias shape:  [hidden_size * 3]

  // hy shape:           [batch_size, hidden_size]
  // workspace shape:    [batch_size, hidden_size * 5]

  std::vector<user_op::OpArg> broadcast_args;
  if (ctx->user_op_conf().has_input("input_bias", 0)) {
    broadcast_args.emplace_back("input_bias", 0);
  }
  if (ctx->user_op_conf().has_input("hidden_bias", 0)) {
    broadcast_args.emplace_back("hidden_bias", 0);
  }

  std::vector<user_op::OpArg> split_args;
  split_args.emplace_back("input_gates", 0);
  split_args.emplace_back("hidden_gates", 0);
  split_args.emplace_back("hx", 0);
  split_args.emplace_back("hy", 0);
  split_args.emplace_back("workspace", 0);

  ctx->NewBuilder().Split(split_args, 0).Broadcast(broadcast_args).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedGruCellOp::InferDataType(user_op::InferContext* ctx) {
  DataType in_types = ctx->InputDType("hx", 0);
  *ctx->MutOutputDType("hy", 0) = in_types;
  *ctx->MutOutputDType("workspace", 0) = in_types;
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedGruCellGradOp ::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& grad_hy_shape = ctx->InputShape("grad_hy", 0);
  DimVector dim_vec({grad_hy_shape.At(0), grad_hy_shape.At(1) * 3});
  *ctx->MutOutputShape("grad_input_gates", 0) = Shape(dim_vec);
  *ctx->MutOutputShape("grad_hidden_gates", 0) = Shape(dim_vec);

  if (ctx->has_output("grad_hx", 0)) { *ctx->MutOutputShape("grad_hx", 0) = grad_hy_shape; }

  if (ctx->has_output("grad_input_bias", 0) && ctx->has_output("grad_hidden_bias", 0)) {
    *ctx->MutOutputShape("grad_input_bias", 0) = Shape({grad_hy_shape.At(1) * 3});
    *ctx->MutOutputShape("grad_hidden_bias", 0) = Shape({grad_hy_shape.At(1) * 3});
  }

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedGruCellGradOp ::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedGruCellGradOp ::GetSbp(user_op::SbpContext* ctx) {
  // grad_hy shape:       [batch_size, hidden_size]
  // workspace shape:     [batch_size, hidden_size * 5]

  // grad_input_gates shape:     [batch_size, hidden_size * 3]
  // grad_hidden_gates shape:    [batch_size, hidden_size * 3]
  // grad_hx shape:              [batch_size, hidden_size]
  // grad_input_bias shape:      [hidden_size * 3]
  // grad_hidden_bias shape:     [hidden_size * 3]

  std::vector<user_op::OpArg> partial_sum_args;
  if (ctx->user_op_conf().has_output("grad_input_bias", 0)) {
    partial_sum_args.emplace_back("grad_input_bias", 0);
  }
  if (ctx->user_op_conf().has_output("grad_hidden_bias", 0)) {
    partial_sum_args.emplace_back("grad_hidden_bias", 0);
  }

  std::vector<user_op::OpArg> split_args;
  split_args.emplace_back("grad_hy", 0);
  split_args.emplace_back("workspace", 0);
  split_args.emplace_back("grad_input_gates", 0);
  split_args.emplace_back("grad_hidden_gates", 0);

  if (ctx->user_op_conf().has_output("grad_hx", 0)) { split_args.emplace_back("grad_hx", 0); }

  ctx->NewBuilder().Split(split_args, 0).PartialSum(partial_sum_args).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedGruCellGradOp ::InferDataType(user_op::InferContext* ctx) {
  DataType in_types = ctx->InputDType("grad_hy", 0);
  *ctx->MutOutputDType("grad_input_gates", 0) = in_types;
  *ctx->MutOutputDType("grad_hidden_gates", 0) = in_types;
  if (ctx->has_output("grad_hx", 0)) { *ctx->MutOutputDType("grad_hx", 0) = in_types; }
  if (ctx->has_output("grad_input_bias", 0)) {
    *ctx->MutOutputDType("grad_input_bias", 0) = in_types;
  }
  if (ctx->has_output("grad_hidden_bias", 0)) {
    *ctx->MutOutputDType("grad_hidden_bias", 0) = in_types;
  }
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("fused_gru_cell")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) -> Maybe<void> {
      const auto grad_op_name = ctx->FwOp().op_name() + "_grad";
      ctx->DefineOp(grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        builder.OpTypeName("fused_gru_cell_grad")
            .InputBind("grad_hy", ctx->FwOp().output_grad("hy", 0))
            .InputBind("workspace", ctx->FwOp().output("workspace", 0))
            .Output("grad_input_gates")
            .Output("grad_hidden_gates");

        if (ctx->FwOp().NeedGenGradTensor4OpInput("hx", 0)) { builder.Output("grad_hx"); }

        if (ctx->FwOp().user_op_conf().has_input("input_bias", 0)
            && ctx->FwOp().user_op_conf().has_input("hidden_bias", 0)) {
          builder.Output("grad_input_bias");
          builder.Output("grad_hidden_bias");
        }
        return builder.Build();
      });

      ctx->FwOp().InputGradBind(user_op::OpArg("input_gates", 0),
                                [&ctx, &grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(grad_op_name).output("grad_input_gates", 0);
                                });
      ctx->FwOp().InputGradBind(user_op::OpArg("hidden_gates", 0),
                                [&ctx, &grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(grad_op_name).output("grad_hidden_gates", 0);
                                });

      if (ctx->FwOp().NeedGenGradTensor4OpInput("hx", 0)) {
        ctx->FwOp().InputGradBind(user_op::OpArg("hx", 0),
                                  [&ctx, &grad_op_name]() -> const std::string& {
                                    return ctx->GetOp(grad_op_name).output("grad_hx", 0);
                                  });
      }

      if (ctx->FwOp().user_op_conf().has_input("input_bias", 0)
          && ctx->FwOp().user_op_conf().has_input("hidden_bias", 0)) {
        ctx->FwOp().InputGradBind(user_op::OpArg("input_bias", 0),
                                  [&ctx, &grad_op_name]() -> const std::string& {
                                    return ctx->GetOp(grad_op_name).output("grad_input_bias", 0);
                                  });
        ctx->FwOp().InputGradBind(user_op::OpArg("hidden_bias", 0),
                                  [&ctx, &grad_op_name]() -> const std::string& {
                                    return ctx->GetOp(grad_op_name).output("grad_hidden_bias", 0);
                                  });
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
