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

namespace oneflow {

namespace user_op {

namespace {
Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const TensorDesc* input =
      ctx->has_input("input", 0) ? &ctx->InputTensorDesc("input", 0) : nullptr;
  const TensorDesc& index = ctx->InputTensorDesc("index", 0);
  const TensorDesc* like = ctx->has_input("like", 0) ? &ctx->InputTensorDesc("like", 0) : nullptr;
  const TensorDesc& src = ctx->InputTensorDesc("src", 0);

  int32_t dim = ctx->Attr<int32_t>("dim");

  // check index.numaxes == src.num_axes == input/like.numaxes
  int64_t src_num_axes = src.shape().NumAxes();
  CHECK_GT_OR_RETURN(src_num_axes, 0);
  CHECK_LE_OR_RETURN(src_num_axes, kDimGatherMaxDimCount);
  int64_t index_num_axes = index.shape().NumAxes();
  CHECK_EQ_OR_RETURN(src_num_axes, index_num_axes);

  int64_t output_num_axes = 0;
  if (input) {
    output_num_axes = input->shape().NumAxes();
  } else if (like) {
    output_num_axes = like->shape().NumAxes();
  } else {
    throw Error::UnimplementedError();
  }
  CHECK_EQ_OR_RETURN(output_num_axes, index_num_axes);

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

  user_op::TensorDesc* out = ctx->OutputTensorDesc("output", 0);
  *out->mut_shape() = input ? input->shape() : like->shape();
  return Maybe<void>::Ok();
}

Maybe<void> InferScalarTensorDesc(user_op::InferContext* ctx) {
  const TensorDesc& input = ctx->InputTensorDesc("input", 0);
  const TensorDesc& index = ctx->InputTensorDesc("index", 0);

  int32_t dim = ctx->Attr<int32_t>("dim");

  // check index.numaxes == src.num_axes == input/like.numaxes
  int64_t output_num_axes = input.shape().NumAxes();
  int64_t index_num_axes = index.shape().NumAxes();
  CHECK_EQ_OR_RETURN(output_num_axes, index_num_axes);

  // check index.shape(i) <= input/like.shape(i)
  FOR_RANGE(int64_t, i, 0, index_num_axes) {
    if (i == dim) continue;
    CHECK_LE_OR_RETURN(index.shape().At(i), input.shape().At(i));
  }

  TensorDesc* out = ctx->OutputTensorDesc("output", 0);
  *out->mut_shape() = input.shape();
  return Maybe<void>::Ok();
}

Maybe<void> InputArgModifierFn(user_op::GetInputArgModifier GetInputArgModifierFn,
                               const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("index", 0);
  CHECK(indices_modifier != nullptr);
  indices_modifier->set_requires_grad(false);

  return Maybe<void>::Ok();
}

Maybe<void> InputScalarArgModifierFn(user_op::GetInputArgModifier GetInputArgModifierFn,
                                     const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("index", 0);
  CHECK(indices_modifier != nullptr);
  indices_modifier->set_requires_grad(false);

  return Maybe<void>::Ok();
}

void _SetSbp(user_op::SbpContext* ctx, const char* like_or_input) {
  const user_op::TensorDesc& index_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("index", 0);
  int64_t index_num_axes = index_tensor.shape().NumAxes();
  const int32_t dim = ctx->Attr<int32_t>("dim");

  FOR_RANGE(int64_t, i, 0, index_num_axes) {
    if (i != dim) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("index", 0), i)
          .Split(user_op::OpArg("src", 0), i)
          .Split(user_op::OpArg("output", 0), i)
          .Split(user_op::OpArg(like_or_input, 0), i)
          .Build();
    } else {
      ctx->NewBuilder()
          .Split(user_op::OpArg("index", 0), i)
          .Split(user_op::OpArg("src", 0), i)
          .PartialSum(user_op::OpArg("output", 0))
          .Broadcast(user_op::OpArg(like_or_input, 0))
          .Build();

      ctx->NewBuilder()
          .Split(user_op::OpArg("index", 0), i)
          .Split(user_op::OpArg("src", 0), i)
          .PartialSum(user_op::OpArg("output", 0))
          .PartialSum(user_op::OpArg(like_or_input, 0))
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

Maybe<void> InferDtype(user_op::InferContext* ctx) {
  const TensorDesc& index = ctx->InputTensorDesc("index", 0);
  CHECK_OR_RETURN(IsIndexDataType(index.data_type()));
  if (ctx->has_input("input", 0)) {
    const TensorDesc& input = ctx->InputTensorDesc("input", 0);
    CHECK_EQ_OR_RETURN(ctx->InputDType("input", 0), ctx->InputDType("src", 0));
  } else {
    CHECK_EQ_OR_RETURN(ctx->InputDType("like", 0), ctx->InputDType("src", 0));
  }
  *ctx->OutputDType("output", 0) = ctx->InputDType("src", 0);
  return Maybe<void>::Ok();
}

Maybe<void> InferScalarDtype(user_op::InferContext* ctx) {
  const TensorDesc& index = ctx->InputTensorDesc("index", 0);
  CHECK_OR_RETURN(IsIndexDataType(index.data_type()));
  *ctx->OutputDType("output", 0) = ctx->InputDType("input", 0);
  return Maybe<void>::Ok();
}

Maybe<void> ScatterBackward(user_op::BackwardOpConfContext* ctx) {
  const TensorDesc& src = ctx->FwOp().TensorDesc4ArgNameAndIndex("src", 0);
  const TensorDesc& index = ctx->FwOp().TensorDesc4ArgNameAndIndex("index", 0);
  const int64_t ndim = src.shape().NumAxes();

  FOR_RANGE(int64_t, i, 0, ndim) {
    if (index.shape().At(i) != src.shape().At(i)) {
      UNIMPLEMENTED() << "The backward pass is implemented only for src.shape == index.shape.\n";
    }
  }

  const auto op_src_grad_name = ctx->FwOp().op_name() + "_src_grad";
  ctx->DefineOp(op_src_grad_name, [&ctx](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("dim_gather")
        .InputBind("index", ctx->FwOp().input("index", 0))
        .InputBind("input", ctx->FwOp().output_grad("output", 0))
        .Output("output")
        .Attr("dim", ctx->FwOp().attr<int32_t>("dim"))
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("src", 0),
                            [&ctx, &op_src_grad_name]() -> const std::string& {
                              return ctx->GetOp(op_src_grad_name).output("output", 0);
                            });
  const auto op_input_grad_name = ctx->FwOp().op_name() + "_input_grad";
  ctx->DefineOp(op_input_grad_name, [&ctx](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("dim_scatter_update_scalar")
        .InputBind("index", ctx->FwOp().input("index", 0))
        .InputBind("input", ctx->FwOp().output_grad("output", 0))
        .Output("output")
        .Attr("dim", ctx->FwOp().attr<int32_t>("dim"))
        .Attr("src_scalar", static_cast<float>(0.0))
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("input", 0),
                            [&ctx, &op_input_grad_name]() -> const std::string& {
                              return ctx->GetOp(op_input_grad_name).output("output", 0);
                            });
  return Maybe<void>::Ok();
}

}  // namespace

#define REGISTER_SCATTER_LIKE_OP(optypename)   \
  REGISTER_USER_OP(optypename)                 \
      .Input("like")                           \
      .Input("index")                          \
      .Input("src")                            \
      .Output("output")                        \
      .Attr<int32_t>("dim")                    \
      .SetTensorDescInferFn(InferTensorDesc)   \
      .SetInputArgModifyFn(InputArgModifierFn) \
      .SetDataTypeInferFn(InferDtype)          \
      .SetGetSbpFn(SetSbpLike)

#define REGISTER_SCATTER_OP(optypename)        \
  REGISTER_USER_OP(optypename)                 \
      .Input("input")                          \
      .Input("index")                          \
      .Input("src")                            \
      .Output("output")                        \
      .Attr<int32_t>("dim")                    \
      .SetTensorDescInferFn(InferTensorDesc)   \
      .SetInputArgModifyFn(InputArgModifierFn) \
      .SetDataTypeInferFn(InferDtype)          \
      .SetGetSbpFn(SetSbpScatter)

#define REGISTER_SCATTER_SCALAR_OP(optypename)       \
  REGISTER_USER_OP(optypename)                       \
      .Input("input")                                \
      .Input("index")                                \
      .Attr<float>("src_scalar")                     \
      .Output("output")                              \
      .Attr<int32_t>("dim")                          \
      .SetTensorDescInferFn(InferScalarTensorDesc)   \
      .SetInputArgModifyFn(InputScalarArgModifierFn) \
      .SetDataTypeInferFn(InferScalarDtype)          \
      .SetGetSbpFn(SetSbpScatter)

#define REGISTER_SCATTER_GRAD(optypename) \
  REGISTER_USER_OP_GRAD(optypename).SetBackwardOpConfGenFn(ScatterBackward);

#define REGISTER_SCATTER_SCALAR_GRAD(optypename)                                               \
  REGISTER_USER_OP_GRAD(optypename)                                                            \
      .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) -> Maybe<void> {         \
        const auto op_input_grad_name = ctx->FwOp().op_name() + "_input_grad";                 \
        ctx->DefineOp(op_input_grad_name, [&ctx](user_op::BackwardOpBuilder& builder) {        \
          return builder.OpTypeName("dim_scatter_update_scalar")                               \
              .InputBind("index", ctx->FwOp().input("index", 0))                               \
              .InputBind("input", ctx->FwOp().output_grad("output", 0))                        \
              .Output("output")                                                                \
              .Attr("dim", ctx->FwOp().attr<int32_t>("dim"))                                   \
              .Attr("src_scalar", static_cast<float>(0.0))                                     \
              .Build();                                                                        \
        });                                                                                    \
        ctx->FwOp().InputGradBind(user_op::OpArg("input", 0),                                  \
                                  [&ctx, &op_input_grad_name]() -> const std::string& {        \
                                    return ctx->GetOp(op_input_grad_name).output("output", 0); \
                                  });                                                          \
        return Maybe<void>::Ok();                                                              \
      });

REGISTER_SCATTER_LIKE_OP("dim_scatter_add_like");
REGISTER_SCATTER_OP("dim_scatter_add");
REGISTER_SCATTER_OP("dim_scatter_update");
REGISTER_SCATTER_OP("dim_scatter_mul");

REGISTER_SCATTER_SCALAR_OP("dim_scatter_update_scalar");
REGISTER_SCATTER_SCALAR_OP("dim_scatter_add_scalar");
REGISTER_SCATTER_SCALAR_OP("dim_scatter_mul_scalar");

REGISTER_SCATTER_GRAD("dim_scatter_add");
REGISTER_SCATTER_GRAD("dim_scatter_update");

REGISTER_SCATTER_SCALAR_GRAD("dim_scatter_update_scalar");
}  // namespace user_op
}  // namespace oneflow
