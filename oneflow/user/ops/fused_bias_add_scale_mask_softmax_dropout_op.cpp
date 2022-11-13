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

namespace {

bool CheckBroadcastable(const Shape& shape, const Shape& broadcast_shape) {
  int left_pad = broadcast_shape.size() - shape.size();
  if (left_pad < 0) { return false; }
  for (int i = 0; i < shape.size(); ++i) {
    int j = i + left_pad;
    if (shape[i] != 1 && shape[i] != broadcast_shape[j]) { return false; }
  }
  return true;
}

bool CheckBroadcastAndSimplifyDims(const Shape& shape, const Shape& broadcast_shape,
                                   int& simplified_ndim, int64_t* simplified_dims) {
  int lpad = broadcast_shape.size() - shape.size();
  if (lpad < 0) { return false; }
  simplified_ndim = 0;
  bool prev_broadcast = false;
  for (int i = 0; i < broadcast_shape.size(); ++i) {
    int64_t dim = (i < lpad) ? 1 : shape[i - lpad];
    int64_t broadcast_dim = broadcast_shape[i];
    if (dim != 1 && dim != broadcast_dim) { return false; }
    bool broadcast = (dim == 1 && broadcast_dim != 1);
    if (simplified_ndim > 0 && broadcast == prev_broadcast) {
      // fold to prev dim
      simplified_dims[simplified_ndim - 1] *= dim;
    } else {
      simplified_dims[simplified_ndim] = dim;
      simplified_ndim += 1;
    }
    prev_broadcast = broadcast;
  }
  return true;
}

// return lpad
int GetBroadcastDims(const Shape& shape, const Shape& broadcast_shape,
                     HashSet<int>& broadcast_dims) {
  int lpad = broadcast_shape.size() - shape.size();
  if (lpad < 0) { return lpad; }
  for (int i = 0; i < broadcast_shape.size(); ++i) {
    if (i < lpad) {
      broadcast_dims.insert(i);
    } else {
      int j = i - lpad;
      if (shape[j] == 1 && shape[j] != broadcast_shape[i]) { broadcast_dims.insert(i); }
      if (shape[j] != 1 && shape[j] != broadcast_shape[i]) { return -1; }
    }
  }
  return lpad;
}

}  // namespace

Maybe<void> FusedBiasAddScaleMaskSoftmaxDropoutOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& bias_shape = ctx->InputShape("bias", 0);
  const Shape& mask_shape = ctx->InputShape("mask", 0);
  const Shape& dropout_mask_shape = ctx->InputShape("dropout_mask", 0);

  CHECK_GE_OR_RETURN(x_shape.size(), 2) << Error::RuntimeError() << "x has at least 2 dimensions";
  CHECK_EQ_OR_RETURN(x_shape.back(), mask_shape.back())
      << " Last dimension of x and mask should be equal, which is softmax dimension.";
  CHECK_EQ_OR_RETURN(dropout_mask_shape, x_shape)
      << Error::RuntimeError() << "dropout_mask shape " << dropout_mask_shape.ToString()
      << " should be equal to x shape " << x_shape.ToString();

  int simplified_bias_ndim = 0;
  int simplified_mask_ndim = 0;
  DimVector simplified_bias_dims(x_shape.size());
  DimVector simplified_mask_dims(x_shape.size());
  CHECK_OR_RETURN(CheckBroadcastAndSimplifyDims(bias_shape, x_shape, simplified_bias_ndim,
                                                simplified_bias_dims.data()))
      << Error::RuntimeError() << "bias shape " << bias_shape.ToString()
      << " could not be broadcast to x shape " << x_shape.ToString();
  CHECK_OR_RETURN(CheckBroadcastAndSimplifyDims(mask_shape, x_shape, simplified_mask_ndim,
                                                simplified_mask_dims.data()))
      << Error::RuntimeError() << "mask shape " << mask_shape.ToString()
      << " could not be broadcast to x shape " << x_shape.ToString();
  CHECK_GT_OR_RETURN(simplified_bias_ndim, 0);  // NOLINT(maybe-need-error-msg)
  CHECK_GT_OR_RETURN(simplified_mask_ndim, 0);  // NOLINT(maybe-need-error-msg)
  // (1, ) -> (K, )
  // (M, 1) -> (M, N)
  // (1, N) -> (M, N)
  // (M, 1, N) -> (M, K, N)
  if ((simplified_bias_ndim == 2 && simplified_bias_dims[0] != 1) || simplified_bias_ndim > 2) {
    return Error::RuntimeError()
           << "bias only support (1, N)->(M, N) broadcast, but got bias shape "
           << bias_shape.ToString() << " broadcast to x shape " << x_shape.ToString();
  }

  if (simplified_mask_ndim > 3 || (simplified_mask_ndim == 3 && simplified_mask_dims[1] != 1)) {
    return Error::RuntimeError() << "mask support (M, 1)->(M, N) or (1, N)->(M, N) or (M, 1, "
                                    "N)->(M, K, N) broadcast, but got mask shape "
                                 << mask_shape.ToString() << " broadcast to x shape "
                                 << x_shape.ToString();
  }

  ctx->SetOutputShape("y", 0, x_shape);
  ctx->SetOutputShape("softmax_y", 0, x_shape);
  ctx->SetOutputIsDynamic("y", 0, ctx->InputIsDynamic("x", 0));
  ctx->SetOutputIsDynamic("softmax_y", 0, ctx->InputIsDynamic("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> FusedBiasAddScaleMaskSoftmaxDropoutOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedBiasAddScaleMaskSoftmaxDropoutOp::InferDataType(user_op::InferContext* ctx) {
  const DataType x_dtype = ctx->InputDType("x", 0);
  const DataType bias_dtype = ctx->InputDType("bias", 0);
  const DataType mask_dtype = ctx->InputDType("mask", 0);
  const DataType dropout_mask_dtype = ctx->InputDType("dropout_mask", 0);

  CHECK_EQ_OR_RETURN(bias_dtype, x_dtype)
      << Error::RuntimeError() << "Expected bias data type " << DataType_Name(x_dtype)
      << ", but got " << DataType_Name(bias_dtype);
  CHECK_OR_RETURN(IsBoolDataType(mask_dtype) || IsIntegralDataType(mask_dtype))
      << Error::RuntimeError() << "Expected mask data type to be bool or integer, but got "
      << DataType_Name(mask_dtype);
  CHECK_OR_RETURN(IsBoolDataType(dropout_mask_dtype))
      << Error::RuntimeError() << "Expected dropout_mask data type to be bool, but got "
      << DataType_Name(dropout_mask_dtype);

  ctx->SetOutputDType("y", 0, x_dtype);
  ctx->SetOutputDType("softmax_y", 0, x_dtype);
  return Maybe<void>::Ok();
}

Maybe<void> FusedBiasAddScaleMaskSoftmaxDropoutOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  const Shape& bias_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("bias", 0).shape();
  const Shape& mask_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("mask", 0).shape();
  const Shape& dropout_mask_shape =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("dropout_mask", 0).shape();

  CHECK_GE_OR_RETURN(x_shape.size(), 2) << Error::RuntimeError() << "x has at least 2 dimensions";
  CHECK_EQ_OR_RETURN(dropout_mask_shape, x_shape)
      << Error::RuntimeError() << "dropout_mask_shape shape " << dropout_mask_shape.ToString()
      << " should be equal to x shape " << x_shape.ToString();

  HashSet<int> bias_broadcast_dims;
  HashSet<int> mask_broadcast_dims;
  int bias_lpad = GetBroadcastDims(bias_shape, x_shape, bias_broadcast_dims);
  int mask_lpad = GetBroadcastDims(mask_shape, x_shape, mask_broadcast_dims);

  CHECK_GE_OR_RETURN(bias_lpad, 0)
      << Error::RuntimeError() << "bias shape " << bias_shape.ToString()
      << " could not be broadcast to x shape " << x_shape.ToString();
  CHECK_GE_OR_RETURN(mask_lpad, 0)
      << Error::RuntimeError() << "mask shape " << mask_shape.ToString()
      << " could not be broadcast to x shape " << x_shape.ToString();

  std::vector<user_op::OpArg> split_args = {
      {"x", 0},
      {"dropout_mask", 0},
      {"y", 0},
      {"softmax_y", 0},
  };

  for (int i = 0; i < x_shape.size(); ++i) {
    bool bias_can_split = (bias_broadcast_dims.find(i) == bias_broadcast_dims.end());
    bool mask_can_split = (mask_broadcast_dims.find(i) == mask_broadcast_dims.end());
    if (bias_can_split && mask_can_split) {
      CHECK_GE_OR_RETURN(i, bias_lpad);  // NOLINT(maybe-need-error-msg)
      CHECK_GE_OR_RETURN(i, mask_lpad);  // NOLINT(maybe-need-error-msg)
      ctx->NewBuilder()
          .Split(split_args, i)
          .Split(user_op::OpArg("bias", 0), i - bias_lpad)
          .Split(user_op::OpArg("mask", 0), i - mask_lpad)
          .Build();
    } else if (bias_can_split) {
      CHECK_GE_OR_RETURN(i, bias_lpad);  // NOLINT(maybe-need-error-msg)
      ctx->NewBuilder()
          .Split(split_args, i)
          .Split(user_op::OpArg("bias", 0), i - bias_lpad)
          .Broadcast(user_op::OpArg("mask", 0))
          .Build();
    } else if (mask_can_split) {
      CHECK_GE_OR_RETURN(i, mask_lpad);  // NOLINT(maybe-need-error-msg)
      ctx->NewBuilder()
          .Split(split_args, i)
          .Broadcast(user_op::OpArg("bias", 0))
          .Split(user_op::OpArg("mask", 0), i - mask_lpad)
          .Build();
    } else {
      ctx->NewBuilder()
          .Split(split_args, i)
          .Broadcast(user_op::OpArg("bias", 0))
          .Broadcast(user_op::OpArg("mask", 0))
          .Build();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> FusedBiasAddScaleMaskSoftmaxDropoutOp::ModifyInputArg(
    const user_op::GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* mask_modifier = GetInputArgModifierFn("mask", 0);
  user_op::InputArgModifier* dropout_mask_modifier = GetInputArgModifierFn("dropout_mask", 0);
  CHECK_OR_RETURN(mask_modifier != nullptr) << " cannot find mask input.";
  CHECK_OR_RETURN(dropout_mask_modifier != nullptr) << " cannot find dropout mask input.";
  mask_modifier->set_requires_grad(false);
  dropout_mask_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
