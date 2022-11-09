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
#include "oneflow/core/common/error.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/user/kernels/dim_scatter_kernel_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {
Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc* input =
      ctx->has_input("input", 0) ? &ctx->InputTensorDesc("input", 0) : nullptr;
  const user_op::TensorDesc& index = ctx->InputTensorDesc("index", 0);
  const user_op::TensorDesc* like =
      ctx->has_input("like", 0) ? &ctx->InputTensorDesc("like", 0) : nullptr;
  const user_op::TensorDesc& src = ctx->InputTensorDesc("src", 0);

  int32_t dim = ctx->Attr<int32_t>("dim");

  // check index.numaxes == src.num_axes == input/like.numaxes
  int64_t src_num_axes = src.shape().NumAxes();
  // For 0-dim Tensor
  CHECK_GE_OR_RETURN(src_num_axes, 0);  // NOLINT
  CHECK_LE_OR_RETURN(src_num_axes, user_op::kDimGatherMaxDimCount);
  int64_t index_num_axes = index.shape().NumAxes();
  CHECK_EQ_OR_RETURN(src_num_axes, index_num_axes);

  int64_t output_num_axes = 0;
  if (input) {
    output_num_axes = input->shape().NumAxes();
  } else if (like) {
    output_num_axes = like->shape().NumAxes();
  } else {
    OF_UNIMPLEMENTED() << "Input tensor and like tensor cannot be empty simultaneously.";
  }
  // For 0-dim Tensor
  if (output_num_axes != 0 && index_num_axes != 0) {
    CHECK_EQ_OR_RETURN(output_num_axes, index_num_axes);  // NOLINT
  } else if (output_num_axes != 0) {
    CHECK_LE_OR_RETURN(output_num_axes, 1);  // NOLINT
  } else {
    CHECK_LE_OR_RETURN(index_num_axes, 1);  // NOLINT
  }

  // check index.shape(i) <= input/like.shape(i)
  FOR_RANGE(int64_t, i, 0, index_num_axes) {
    if (i == dim) continue;
    if (input) {
      CHECK_LE_OR_RETURN(index.shape().At(i), input->shape().At(i));
    } else {
      CHECK_LE_OR_RETURN(index.shape().At(i), like->shape().At(i));
    }
  }

  // check index.shape(i) <= src.shape(i)
  FOR_RANGE(int64_t, i, 0, index_num_axes) {
    if (i == dim) continue;
    CHECK_LE_OR_RETURN(index.shape().At(i), src.shape().At(i));
  }

  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("output", 0);
  out->set_shape(input ? input->shape() : like->shape());
  return Maybe<void>::Ok();
}

Maybe<void> InferScalarTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& index = ctx->InputTensorDesc("index", 0);

  int32_t dim = ctx->Attr<int32_t>("dim");

  // check index.numaxes == src.num_axes == input/like.numaxes
  int64_t output_num_axes = input.shape().NumAxes();
  int64_t index_num_axes = index.shape().NumAxes();
  // For 0-dim tensor
  CHECK_GE_OR_RETURN(output_num_axes, index_num_axes);  // NOLINT

  // check index.shape(i) <= input/like.shape(i)
  FOR_RANGE(int64_t, i, 0, index_num_axes) {
    if (i == dim) continue;
    CHECK_LE_OR_RETURN(index.shape().At(i), input.shape().At(i));
  }

  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("output", 0);
  out->set_shape(input.shape());
  return Maybe<void>::Ok();
}

Maybe<void> InputArgModifierFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                               const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("index", 0);
  CHECK_OR_RETURN(indices_modifier != nullptr);
  indices_modifier->set_requires_grad(false);

  return Maybe<void>::Ok();
}

Maybe<void> InputScalarArgModifierFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                     const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("index", 0);
  CHECK_OR_RETURN(indices_modifier != nullptr);
  indices_modifier->set_requires_grad(false);

  return Maybe<void>::Ok();
}

void _SetSbp(user_op::SbpContext* ctx, const char* like_or_input) {
  const int32_t dim = ctx->Attr<int32_t>("dim");

  const Shape& index_tensor_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("index", 0).shape();
  const Shape& src_tensor_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("src", 0).shape();
  const Shape& input_tensor_shape =
      ctx->LogicalTensorDesc4InputArgNameAndIndex(like_or_input, 0).shape();

  FOR_RANGE(int64_t, i, 0, index_tensor_shape.NumAxes()) {
    if (i == dim) { continue; }
    int64_t len = index_tensor_shape.At(i);
    if (len == src_tensor_shape.At(i) && len == input_tensor_shape.At(i)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("index", 0), i)
          .Split(user_op::OpArg("src", 0), i)
          .Split(user_op::OpArg(like_or_input, 0), i)
          .Split(user_op::OpArg("output", 0), i)
          .Build();
    }
  }

  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("src", 0))
      .Broadcast(user_op::OpArg("index", 0))
      .PartialSum(user_op::OpArg("output", 0))
      .PartialSum(user_op::OpArg(like_or_input, 0))
      .Build();
}

Maybe<void> SetSbpLike(user_op::SbpContext* ctx) {
  _SetSbp(ctx, "like");
  return Maybe<void>::Ok();
}

Maybe<void> SetSbpScatter(user_op::SbpContext* ctx) {
  _SetSbp(ctx, "input");
  return Maybe<void>::Ok();
}

Maybe<void> SetSbpScatterScalar(user_op::SbpContext* ctx) {
  const int32_t dim = ctx->Attr<int32_t>("dim");

  const Shape& index_tensor_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("index", 0).shape();
  const Shape& input_tensor_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0).shape();

  FOR_RANGE(int64_t, i, 0, index_tensor_shape.NumAxes()) {
    if (i == dim) { continue; }
    if (index_tensor_shape.At(i) == input_tensor_shape.At(i)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("index", 0), i)
          .Split(user_op::OpArg("input", 0), i)
          .Split(user_op::OpArg("output", 0), i)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferDtype(user_op::InferContext* ctx) {
  const user_op::TensorDesc& index = ctx->InputTensorDesc("index", 0);
  CHECK_OR_RETURN(IsIndexDataType(index.data_type()));
  if (ctx->has_input("input", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("input", 0), ctx->InputDType("src", 0))
        << "InferDataType Failed. Expected " << DataType_Name(ctx->InputDType("src", 0))
        << ", but got " << DataType_Name(ctx->InputDType("input", 0));
  } else {
    CHECK_EQ_OR_RETURN(ctx->InputDType("like", 0), ctx->InputDType("src", 0))
        << "InferDataType Failed. Expected " << DataType_Name(ctx->InputDType("like", 0))
        << ", but got " << DataType_Name(ctx->InputDType("src", 0));
  }
  ctx->SetOutputDType("output", 0, ctx->InputDType("src", 0));
  return Maybe<void>::Ok();
}

Maybe<void> InferScalarDtype(user_op::InferContext* ctx) {
  const user_op::TensorDesc& index = ctx->InputTensorDesc("index", 0);
  CHECK_OR_RETURN(IsIndexDataType(index.data_type()));
  ctx->SetOutputDType("output", 0, ctx->InputDType("input", 0));
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> DimScatterAddLikeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc(ctx);
}

/*static*/ Maybe<void> DimScatterAddLikeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DimScatterAddLikeOp::GetSbp(user_op::SbpContext* ctx) {
  return SetSbpLike(ctx);
}

/* static */ Maybe<void> DimScatterAddLikeOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return InputArgModifierFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> DimScatterAddLikeOp::InferDataType(user_op::InferContext* ctx) {
  return InferDtype(ctx);
}

#define DEF_SCATTER_OP(op_class_name)                                                             \
  /* static */ Maybe<void> op_class_name::InferLogicalTensorDesc(user_op::InferContext* ctx) {    \
    return InferTensorDesc(ctx);                                                                  \
  }                                                                                               \
                                                                                                  \
  /*static*/ Maybe<void> op_class_name::InferPhysicalTensorDesc(user_op::InferContext* ctx) {     \
    return InferLogicalTensorDesc(ctx);                                                           \
  }                                                                                               \
                                                                                                  \
  /* static */ Maybe<void> op_class_name::GetSbp(user_op::SbpContext* ctx) {                      \
    return SetSbpScatter(ctx);                                                                    \
  }                                                                                               \
                                                                                                  \
  /* static */ Maybe<void> op_class_name::ModifyInputArg(                                         \
      const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) { \
    return InputArgModifierFn(GetInputArgModifierFn, conf);                                       \
  }                                                                                               \
                                                                                                  \
  /* static */ Maybe<void> op_class_name::InferDataType(user_op::InferContext* ctx) {             \
    return InferDtype(ctx);                                                                       \
  }

#define DEF_SCATTER_SCALAR_OP(optypename)                                                         \
  /* static */ Maybe<void> optypename::InferLogicalTensorDesc(user_op::InferContext* ctx) {       \
    return InferScalarTensorDesc(ctx);                                                            \
  }                                                                                               \
                                                                                                  \
  /*static*/ Maybe<void> optypename::InferPhysicalTensorDesc(user_op::InferContext* ctx) {        \
    return InferLogicalTensorDesc(ctx);                                                           \
  }                                                                                               \
                                                                                                  \
  /* static */ Maybe<void> optypename::GetSbp(user_op::SbpContext* ctx) {                         \
    return SetSbpScatterScalar(ctx);                                                              \
  }                                                                                               \
                                                                                                  \
  /* static */ Maybe<void> optypename::ModifyInputArg(                                            \
      const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) { \
    return InputScalarArgModifierFn(GetInputArgModifierFn, conf);                                 \
  }                                                                                               \
                                                                                                  \
  /* static */ Maybe<void> optypename::InferDataType(user_op::InferContext* ctx) {                \
    return InferScalarDtype(ctx);                                                                 \
  }

DEF_SCATTER_OP(DimScatterAddOp);
DEF_SCATTER_OP(DimScatterUpdateOp);
DEF_SCATTER_OP(DimScatterMulOp);

DEF_SCATTER_SCALAR_OP(DimScatterUpdateScalarOp);
DEF_SCATTER_SCALAR_OP(DimScatterAddScalarOp);
DEF_SCATTER_SCALAR_OP(DimScatterMulScalarOp);

}  // namespace oneflow
