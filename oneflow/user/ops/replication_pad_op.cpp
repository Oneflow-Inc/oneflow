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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

template<size_t ndim>
Maybe<void> GetOpSbpSignature(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  const int64_t input_dims = x_tensor.shape().NumAxes();
  const int64_t first_two_dims = input_dims - (ndim - 2);
  FOR_RANGE(int64_t, i, 0, first_two_dims) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

template<size_t ndim>
Maybe<void> GetOpGradSbpSignature(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& dy_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0);
  const int64_t grad_dims = dy_tensor.shape().NumAxes();
  CHECK_EQ_OR_RETURN(grad_dims, ndim);  // NOLINT
  const int64_t first_two_dims = grad_dims - (ndim - 2);
  FOR_RANGE(int64_t, i, 0, first_two_dims) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

}  // namespace

/*static*/ Maybe<void> ReplicationPad1DOp::GetSbp(user_op::SbpContext* ctx) {
  return GetOpSbpSignature<3>(ctx);
}
/*static*/ Maybe<void> ReplicationPad1DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");
  const int64_t n_idx = 0;
  const int64_t c_idx = 1;
  const int64_t w_idx = 2;

  DimVector y_dim_vec(x_shape.NumAxes());
  const int64_t w_x = x_shape.At(w_idx);

  y_dim_vec[n_idx] = x_shape.At(n_idx);
  y_dim_vec[c_idx] = x_shape.At(c_idx);
  y_dim_vec[w_idx] = w_x + padding[0] + padding[1];

  ctx->SetOutputShape("y", 0, Shape(y_dim_vec));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ReplicationPad1DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return ReplicationPad1DOp::InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ReplicationPad1DOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ReplicationPad1DOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* x_modifier = GetInputArgModifierFn("x", 0);
  CHECK_NOTNULL_OR_RETURN(x_modifier);
  x_modifier->set_requires_grad(true);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ReplicationPad1DGradOp::GetSbp(user_op::SbpContext* ctx) {
  return GetOpGradSbpSignature<3>(ctx);
}
/*static*/ Maybe<void> ReplicationPad1DGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");
  CHECK_EQ_OR_RETURN(padding.size(), dy_shape.NumAxes() - 1);  // NOLINT
  const int64_t n_idx = 0;
  const int64_t c_idx = 1;
  const int64_t w_idx = 2;

  DimVector dx_dim_vec(dy_shape.NumAxes());
  int64_t w_dy = dy_shape.At(w_idx);

  dx_dim_vec[n_idx] = dy_shape.At(0);
  dx_dim_vec[c_idx] = dy_shape.At(1);
  dx_dim_vec[w_idx] = w_dy - padding[0] - padding[1];

  ctx->SetOutputShape("dx", 0, Shape(dx_dim_vec));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ReplicationPad1DGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return ReplicationPad1DGradOp::InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ReplicationPad1DGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ReplicationPad2DOp::GetSbp(user_op::SbpContext* ctx) {
  return GetOpSbpSignature<4>(ctx);
}
/*static*/ Maybe<void> ReplicationPad2DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");
  CHECK_EQ_OR_RETURN(padding.size(), x_shape.NumAxes());  // NOLINT
  const int64_t n_idx = 0;
  const int64_t c_idx = 1;
  const int64_t h_idx = 2;
  const int64_t w_idx = 3;

  DimVector y_dim_vec(x_shape.NumAxes());
  const int64_t h_x = x_shape.At(h_idx);
  const int64_t w_x = x_shape.At(w_idx);

  y_dim_vec[n_idx] = x_shape.At(n_idx);
  y_dim_vec[c_idx] = x_shape.At(c_idx);
  y_dim_vec[h_idx] = h_x + padding[2] + padding[3];
  y_dim_vec[w_idx] = w_x + padding[0] + padding[1];

  ctx->SetOutputShape("y", 0, Shape(y_dim_vec));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ReplicationPad2DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return ReplicationPad2DOp::InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ReplicationPad2DOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ReplicationPad2DOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* x_modifier = GetInputArgModifierFn("x", 0);
  CHECK_NOTNULL_OR_RETURN(x_modifier);  // NOLINT
  x_modifier->set_requires_grad(true);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ReplicationPad2DGradOp::GetSbp(user_op::SbpContext* ctx) {
  return GetOpGradSbpSignature<4>(ctx);
}
/*static*/ Maybe<void> ReplicationPad2DGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");
  CHECK_EQ_OR_RETURN(padding.size(), dy_shape.NumAxes());  // NOLINT
  const int64_t n_idx = 0;
  const int64_t c_idx = 1;
  const int64_t h_idx = 2;
  const int64_t w_idx = 3;

  DimVector dx_dim_vec(dy_shape.NumAxes());
  int64_t h_dy = dy_shape.At(h_idx);
  int64_t w_dy = dy_shape.At(w_idx);

  dx_dim_vec[n_idx] = dy_shape.At(0);
  dx_dim_vec[c_idx] = dy_shape.At(1);
  dx_dim_vec[h_idx] = h_dy - padding[2] - padding[3];
  dx_dim_vec[w_idx] = w_dy - padding[0] - padding[1];

  ctx->SetOutputShape("dx", 0, Shape(dx_dim_vec));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ReplicationPad2DGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return ReplicationPad2DGradOp::InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ReplicationPad2DGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
