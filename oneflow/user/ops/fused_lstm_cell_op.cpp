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

/* static */ Maybe<void> FusedLstmCellOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& cx_shape = ctx->InputShape("cx", 0);
  ctx->SetOutputShape("hy", 0, cx_shape);
  ctx->SetOutputShape("cy", 0, cx_shape);
  ctx->SetOutputShape("workspace", 0, ctx->InputShape("input_gates", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedLstmCellOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedLstmCellOp::GetSbp(user_op::SbpContext* ctx) {
  // input_gates shape:  [batch_size, hidden_size * 4]
  // hidden_gates shape: [batch_size, hidden_size * 4]
  // cx shape:           [batch_size, hidden_size]
  // input_bias shape:   [hidden_size * 4]
  // hidden_bias shape:  [hidden_size * 4]

  // hy shape:           [batch_size, hidden_size]
  // cy shape:           [batch_size, hidden_size]
  // workspace shape:    [batch_size, hidden_size * 4]

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
  split_args.emplace_back("cx", 0);
  split_args.emplace_back("hy", 0);
  split_args.emplace_back("cy", 0);
  split_args.emplace_back("workspace", 0);

  ctx->NewBuilder().Split(split_args, 0).Broadcast(broadcast_args).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedLstmCellOp::InferDataType(user_op::InferContext* ctx) {
  DataType in_types = ctx->InputDType("cx", 0);
  ctx->SetOutputDType("hy", 0, in_types);
  ctx->SetOutputDType("cy", 0, in_types);
  ctx->SetOutputDType("workspace", 0, in_types);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedLstmCellGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("grad_gates", 0, ctx->InputShape("workspace", 0));

  if (ctx->has_output("grad_cx", 0)) {
    ctx->SetOutputShape("grad_cx", 0, ctx->InputShape("cx", 0));
  }

  if (ctx->has_output("grad_bias", 0)) {
    ctx->SetOutputShape("grad_bias", 0, Shape({ctx->InputShape("workspace", 0).At(1)}));
  }

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedLstmCellGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedLstmCellGradOp::GetSbp(user_op::SbpContext* ctx) {
  // grad_hy shape:       [batch_size, hidden_size]
  // grad_cy shape:       [batch_size, hidden_size]
  // cx shape:            [batch_size, hidden_size]
  // cy shape:            [batch_size, hidden_size]
  // workspace shape:     [batch_size, hidden_size * 4]

  // grad_gates shape:    [batch_size, hidden_size * 4]
  // grad_cx shape:       [batch_size, hidden_size]
  // grad_bias shape:     [hidden_size * 4]

  std::vector<user_op::OpArg> partial_sum_args;
  if (ctx->user_op_conf().has_output("grad_bias", 0)) {
    partial_sum_args.emplace_back("grad_bias", 0);
  }

  std::vector<user_op::OpArg> split_args;
  split_args.emplace_back("grad_hy", 0);
  split_args.emplace_back("grad_cy", 0);
  split_args.emplace_back("cx", 0);
  split_args.emplace_back("cy", 0);
  split_args.emplace_back("workspace", 0);
  split_args.emplace_back("grad_gates", 0);

  if (ctx->user_op_conf().has_output("grad_cx", 0)) { split_args.emplace_back("grad_cx", 0); }

  ctx->NewBuilder().Split(split_args, 0).PartialSum(partial_sum_args).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedLstmCellGradOp::InferDataType(user_op::InferContext* ctx) {
  DataType in_types = ctx->InputDType("grad_hy", 0);
  ctx->SetOutputDType("grad_gates", 0, in_types);
  if (ctx->has_output("grad_cx", 0)) { ctx->SetOutputDType("grad_cx", 0, in_types); }
  if (ctx->has_output("grad_bias", 0)) { ctx->SetOutputDType("grad_bias", 0, in_types); }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
