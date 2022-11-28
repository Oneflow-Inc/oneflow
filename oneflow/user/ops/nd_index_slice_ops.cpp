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

Maybe<void> CheckScatterNdShape(const Shape& params_shape, const Shape& indices_shape,
                                const Shape& updates_shape) {
  int64_t batch_ndims = indices_shape.NumAxes() - 1;
  int64_t index_ndims = indices_shape.At(batch_ndims);
  CHECK_LE_OR_RETURN(batch_ndims, updates_shape.NumAxes());
  CHECK_LE_OR_RETURN(index_ndims, params_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, batch_ndims) {
    CHECK_EQ_OR_RETURN(updates_shape.At(i), indices_shape.At(i));
  }
  int64_t slice_ndims = params_shape.NumAxes() - index_ndims;
  CHECK_EQ_OR_RETURN(slice_ndims, updates_shape.NumAxes() - batch_ndims);
  FOR_RANGE(int64_t, i, 0, slice_ndims) {
    CHECK_EQ_OR_RETURN(updates_shape.At(i + batch_ndims), params_shape.At(i + index_ndims));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferScatterNdTensorDesc(user_op::InferContext* ctx) {
  const Shape& indices_shape = ctx->InputShape("indices", 0);
  const Shape& updates_shape = ctx->InputShape("updates", 0);
  const Shape& params_shape = ctx->Attr<Shape>("shape");
  JUST(CheckScatterNdShape(params_shape, indices_shape, updates_shape));
  ctx->SetOutputShape("out", 0, params_shape);
  return Maybe<void>::Ok();
}

Maybe<void> InferScatterNdDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("updates", 0));
  return Maybe<void>::Ok();
}

Maybe<void> InferScatterNdLikeTensorDesc(user_op::InferContext* ctx) {
  const Shape& indices_shape = ctx->InputShape("indices", 0);
  const Shape& updates_shape = ctx->InputShape("updates", 0);
  const Shape& like_shape = ctx->InputShape("like", 0);
  JUST(CheckScatterNdShape(like_shape, indices_shape, updates_shape));
  ctx->SetOutputShape("out", 0, like_shape);
  return Maybe<void>::Ok();
}

Maybe<void> InferScatterNdLikeDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("updates", 0));
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorScatterNdOptTensorDesc(user_op::InferContext* ctx) {
  const Shape& params_shape = ctx->InputShape("params", 0);
  const Shape& updates_shape = ctx->InputShape("updates", 0);
  const Shape& indices_shape = ctx->InputShape("indices", 0);
  JUST(CheckScatterNdShape(params_shape, indices_shape, updates_shape));
  ctx->SetOutputShape("out", 0, params_shape);
  ctx->SetOutputStride("out", 0, ctx->InputStride("params", 0));
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorScatterNdOptDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("params", 0));
  return Maybe<void>::Ok();
}

Maybe<void> GetTensorScatterNdOptSbpSignatures(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& params_tensor =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("params", 0);
  const user_op::TensorDesc& indices_tensor =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0);
  int64_t indices_num_axes = indices_tensor.shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("params", 0))
        .Split(user_op::OpArg("indices", 0), i)
        .Split(user_op::OpArg("updates", 0), i)
        .Broadcast(user_op::OpArg("out", 0))
        .Build();
  }
  int64_t index_ndims = indices_tensor.shape().At(indices_num_axes - 1);
  FOR_RANGE(int64_t, i, index_ndims, params_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("params", 0), i)
        .Broadcast(user_op::OpArg("indices", 0))
        .Split(user_op::OpArg("updates", 0), i - index_ndims + indices_num_axes - 1)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("params", 0))
      .Broadcast(user_op::OpArg("indices", 0))
      .PartialSum(user_op::OpArg("updates", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> GatherNdOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& params_shape = ctx->InputShape("params", 0);
  const Shape& indices_shape = ctx->InputShape("indices", 0);
  int64_t index_ndims = indices_shape.At(indices_shape.NumAxes() - 1);
  CHECK_LE_OR_RETURN(index_ndims, params_shape.NumAxes());
  DimVector out_shape_vec(indices_shape.dim_vec().cbegin(), indices_shape.dim_vec().cend() - 1);
  FOR_RANGE(int64_t, i, index_ndims, params_shape.NumAxes()) {
    out_shape_vec.emplace_back(params_shape.At(i));
  }
  const Shape& out_shape = Shape(out_shape_vec);
  bool is_out_of_bounds = params_shape.Count(0) == 0 && out_shape.Count(0) != 0;
  CHECK_OR_RETURN(!is_out_of_bounds)
      << Error::IndexError() << "The index is out of bounds for dimension with size 0";
  ctx->SetOutputShape("out", 0, out_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> GatherNdOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> GatherNdOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& params_tensor =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("params", 0);
  const user_op::TensorDesc& indices_tensor =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0);
  int64_t indices_num_axes = indices_tensor.shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("params", 0))
        .Split(user_op::OpArg("indices", 0), i)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }
  int64_t index_ndims = indices_tensor.shape().At(indices_num_axes - 1);
  FOR_RANGE(int64_t, i, index_ndims, params_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("params", 0), i)
        .Broadcast(user_op::OpArg("indices", 0))
        .Split(user_op::OpArg("out", 0), i - index_ndims + indices_num_axes - 1)
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("params", 0))
      .Broadcast(user_op::OpArg("indices", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> GatherNdOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("indices", 0);
  CHECK_OR_RETURN(indices_modifier != nullptr);
  indices_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> GatherNdOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("params", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ScatterNdOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferScatterNdTensorDesc(ctx);
}

/*static*/ Maybe<void> ScatterNdOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ScatterNdOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& indices_desc =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0);
  int64_t indices_num_axes = indices_desc.shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("indices", 0), i)
        .Split(user_op::OpArg("updates", 0), i)
        .Broadcast(user_op::OpArg("out", 0))
        .Build();
  }
  const Shape& out_shape = ctx->Attr<Shape>("shape");
  int64_t index_ndims = indices_desc.shape().At(indices_num_axes - 1);
  int64_t slice_ndims = out_shape.NumAxes() - index_ndims;
  FOR_RANGE(int64_t, i, 0, slice_ndims) {
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("indices", 0))
        .Split(user_op::OpArg("updates", 0), i + indices_num_axes - 1)
        .Split(user_op::OpArg("out", 0), i + index_ndims)
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("updates", 0))
      .Broadcast(user_op::OpArg("indices", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ScatterNdOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("indices", 0);
  CHECK_OR_RETURN(indices_modifier != nullptr);
  indices_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ScatterNdOp::InferDataType(user_op::InferContext* ctx) {
  return InferScatterNdDataType(ctx);
}

/* static */ Maybe<void> ScatterNdLikeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferScatterNdLikeTensorDesc(ctx);
}

/*static*/ Maybe<void> ScatterNdLikeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ScatterNdLikeOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& indices_tensor =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0);
  int64_t indices_num_axes = indices_tensor.shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("like", 0))
        .Split(user_op::OpArg("indices", 0), i)
        .Split(user_op::OpArg("updates", 0), i)
        .Broadcast(user_op::OpArg("out", 0))
        .Build();
  }
  const Shape& out_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape();
  int64_t index_ndims = indices_tensor.shape().At(indices_num_axes - 1);
  int64_t slice_ndims = out_shape.NumAxes() - index_ndims;
  FOR_RANGE(int64_t, i, 0, slice_ndims) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("like", 0), i + index_ndims)
        .Broadcast(user_op::OpArg("indices", 0))
        .Split(user_op::OpArg("updates", 0), i + indices_num_axes - 1)
        .Split(user_op::OpArg("out", 0), i + index_ndims)
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("like", 0))
      .PartialSum(user_op::OpArg("updates", 0))
      .Broadcast(user_op::OpArg("indices", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ScatterNdLikeOp::InferDataType(user_op::InferContext* ctx) {
  return InferScatterNdLikeDataType(ctx);
}

/* static */ Maybe<void> TensorScatterNdUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorScatterNdOptTensorDesc(ctx);
}

/*static*/ Maybe<void> TensorScatterNdUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> TensorScatterNdUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  return GetTensorScatterNdOptSbpSignatures(ctx);
}

/* static */ Maybe<void> TensorScatterNdUpdateOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("indices", 0);
  CHECK_OR_RETURN(indices_modifier != nullptr);
  indices_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> TensorScatterNdUpdateOp::InferDataType(user_op::InferContext* ctx) {
  return InferTensorScatterNdOptDataType(ctx);
}

/* static */ Maybe<void> TensorScatterNdAddOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorScatterNdOptTensorDesc(ctx);
}

/*static*/ Maybe<void> TensorScatterNdAddOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> TensorScatterNdAddOp::GetSbp(user_op::SbpContext* ctx) {
  return GetTensorScatterNdOptSbpSignatures(ctx);
}

/* static */ Maybe<void> TensorScatterNdAddOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("indices", 0);
  CHECK_OR_RETURN(indices_modifier != nullptr);
  indices_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> TensorScatterNdAddOp::InferDataType(user_op::InferContext* ctx) {
  return InferTensorScatterNdOptDataType(ctx);
}

}  // namespace oneflow
