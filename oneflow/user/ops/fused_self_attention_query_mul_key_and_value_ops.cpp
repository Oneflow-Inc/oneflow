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

/*static*/ auto FusedSelfAttentionQueryMulKeyAndValueOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  const DataType& dtype = ctx->InputDType("hidden_states", 0);
  *ctx->OutputDType("query_mul_key", 0) = dtype;
  *ctx->OutputDType("value", 0) = dtype;
  return Maybe<void>::Ok();
}
/*static*/ auto FusedSelfAttentionQueryMulKeyAndValueOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  CHECK_OR_RETURN(!(ctx->InputIsDynamic("hidden_states", 0)));
  int64_t head_size = ctx->Attr<int64_t>("head_size");
  const Shape& hidden_states_shape = ctx->InputShape("hidden_states", 0);
  // hidden_states_shape (seq_len, batch_size, hidden_size)
  // layout is (seq_len, batch_size, num_heads, 3, head_size)
  // for example shape (1024, 4, 12, 3, 64) -> (1024, 4, 12, 192) which stride is (9216, 2304,
  // 192, 1)
  CHECK_EQ_OR_RETURN(hidden_states_shape.NumAxes(), 3);
  int64_t seq_len = hidden_states_shape.At(0);
  int64_t batch_size = hidden_states_shape.At(1);
  int64_t hidden_size = hidden_states_shape.At(2);
  CHECK_EQ_OR_RETURN(hidden_size % (head_size * 3), 0);
  int64_t num_heads = hidden_size / (head_size * 3);

  *ctx->MutOutputShape("query_mul_key", 0) = Shape({batch_size, num_heads, seq_len, seq_len});
  *ctx->MutOutputShape("value", 0) = Shape({batch_size, num_heads, seq_len, head_size});

  return Maybe<void>::Ok();
}
/*static*/ auto FusedSelfAttentionQueryMulKeyAndValueOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  return FusedSelfAttentionQueryMulKeyAndValueOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto FusedSelfAttentionQueryMulKeyAndValueOp::GetSbp(user_op::SbpContext* ctx)
    -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("hidden_states", 0), 1)
      .Split(user_op::OpArg("query_mul_key", 0), 0)
      .Split(user_op::OpArg("value", 0), 0)
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("hidden_states", 0), 2)
      .Split(user_op::OpArg("query_mul_key", 0), 1)
      .Split(user_op::OpArg("value", 0), 1)
      .Build();
  return Maybe<void>::Ok();
}

/*static*/ auto FusedSelfAttentionQueryMulKeyAndValueGradOp::InferDataType(
    user_op::InferContext* ctx) -> Maybe<void> {
  const DataType& dtype = ctx->InputDType("query_mul_key_grad", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("value_grad", 0), dtype);
  *ctx->OutputDType("hidden_states_grad", 0) = dtype;
  return Maybe<void>::Ok();
}
/*static*/ auto FusedSelfAttentionQueryMulKeyAndValueGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  CHECK_OR_RETURN(!(ctx->InputIsDynamic("query_mul_key_grad", 0)));
  CHECK_OR_RETURN(!(ctx->InputIsDynamic("value_grad", 0)));
  const Shape& h_shape = ctx->InputShape("hidden_states", 0);
  const Shape& qmk_grad_shape = ctx->InputShape("query_mul_key_grad", 0);
  const Shape& v_grad_shape = ctx->InputShape("value_grad", 0);
  CHECK_EQ_OR_RETURN(h_shape.NumAxes(), 3);
  CHECK_EQ_OR_RETURN(qmk_grad_shape.NumAxes(), 4);
  CHECK_EQ_OR_RETURN(v_grad_shape.NumAxes(), 4);
  // hidden_states shape (s, b, H)
  int64_t seq_len = h_shape.At(0);
  int64_t batch_size = h_shape.At(1);
  int64_t hidden_size = h_shape.At(2);
  // value grad shape (b, n, s, h)
  int64_t num_heads = v_grad_shape.At(1);
  int64_t head_size = v_grad_shape.At(3);
  CHECK_EQ_OR_RETURN(v_grad_shape.At(0), batch_size);
  CHECK_EQ_OR_RETURN(v_grad_shape.At(2), seq_len);
  CHECK_EQ_OR_RETURN(hidden_size, num_heads * 3 * head_size);
  // qmk grad shape (b, n, sq, sk)
  CHECK_EQ_OR_RETURN(qmk_grad_shape.At(0), batch_size);
  CHECK_EQ_OR_RETURN(qmk_grad_shape.At(1), num_heads);
  CHECK_EQ_OR_RETURN(qmk_grad_shape.At(2), seq_len);
  CHECK_EQ_OR_RETURN(qmk_grad_shape.At(3), seq_len);

  *ctx->MutOutputShape("hidden_states_grad", 0) = h_shape;
  return Maybe<void>::Ok();
}
/*static*/ auto FusedSelfAttentionQueryMulKeyAndValueGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  return FusedSelfAttentionQueryMulKeyAndValueGradOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto FusedSelfAttentionQueryMulKeyAndValueGradOp::GetSbp(user_op::SbpContext* ctx)
    -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("query_mul_key_grad", 0), 0)
      .Split(user_op::OpArg("value_grad", 0), 0)
      .Split(user_op::OpArg("hidden_states", 0), 1)
      .Split(user_op::OpArg("hidden_states_grad", 0), 1)
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("query_mul_key_grad", 0), 1)
      .Split(user_op::OpArg("value_grad", 0), 1)
      .Split(user_op::OpArg("hidden_states", 0), 2)
      .Split(user_op::OpArg("hidden_states_grad", 0), 2)
      .Build();
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("fused_self_attention_query_mul_key_and_value")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) -> Maybe<void> {
      std::string grad_op_name = ctx->FwOp().op_name() + "_grad";

      ctx->DefineOp(grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("fused_self_attention_query_mul_key_and_value_grad")
            .InputBind("hidden_states", ctx->FwOp().input("hidden_states", 0))
            .InputBind("query_mul_key_grad", ctx->FwOp().output_grad("query_mul_key", 0))
            .InputBind("value_grad", ctx->FwOp().output_grad("value", 0))
            .Output("hidden_states_grad")
            .Attr("alpha", ctx->FwOp().attr<float>("alpha"))
            .Build();
      });

      ctx->FwOp().InputGradBind(user_op::OpArg("hidden_states", 0), [=]() -> const std::string& {
        return ctx->GetOp(grad_op_name).output("hidden_states_grad", 0);
      });
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
