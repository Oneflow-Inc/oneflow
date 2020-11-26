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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/user/kernels/dim_gather_scatter_util.h"

namespace oneflow {

namespace user_op {

namespace {
Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const TensorDesc* input = ctx->TensorDesc4ArgNameAndIndex("input", 0);
  const TensorDesc* index = ctx->TensorDesc4ArgNameAndIndex("index", 0);
  const TensorDesc* like = ctx->TensorDesc4ArgNameAndIndex("like", 0);

  const Shape& like_shape = like->shape();
  int32_t dim = ctx->Attr<int32_t>("dim");

  const SbpParallel& input_sbp = ctx->SbpParallel4ArgNameAndIndex("input", 0);
  int64_t split_axis = input_sbp.split_parallel().axis();
  if (ctx->parallel_ctx().parallel_num() != 1 && input_sbp.has_split_parallel()) {
    CHECK_NE_OR_RETURN(split_axis, dim) << "split_axis should NOT equal dim";
  }

  int64_t input_num_axes = input->shape().NumAxes();
  CHECK_GT_OR_RETURN(input_num_axes, 0);
  CHECK_LE_OR_RETURN(input_num_axes, kDimGatherMaxDimCount);

  int64_t index_num_axes = index->shape().NumAxes();
  CHECK_EQ_OR_RETURN(input_num_axes, index_num_axes);
  CHECK_EQ_OR_RETURN(input_num_axes, like_shape.NumAxes());

  FOR_RANGE(int64_t, i, 0, input_num_axes) {
    CHECK_EQ_OR_RETURN(index->shape().At(i), input->shape().At(i));
  }

  user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("output", 0);
  *out->mut_shape() = like_shape;
  *out->mut_data_type() = input->data_type();

  return Maybe<void>::Ok();
}

Maybe<void> InputArgModifierFn(user_op::GetInputArgModifier GetInputArgModifierFn,
                               const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* like_arg_modifier = GetInputArgModifierFn("like", 0);
  CHECK(like_arg_modifier != nullptr);
  like_arg_modifier->set_use_header_only(true);
  like_arg_modifier->set_requires_grad(false);

  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("index", 0);
  CHECK(indices_modifier != nullptr);
  indices_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

Maybe<void> InplaceInputArgModifierFn(user_op::GetInputArgModifier GetInputArgModifierFn,
                                      const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* like_arg_modifier = GetInputArgModifierFn("like", 0);
  CHECK(like_arg_modifier != nullptr);
  like_arg_modifier->set_requires_grad(false);

  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("index", 0);
  CHECK(indices_modifier != nullptr);
  indices_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

Maybe<void> InferBatchAxis(user_op::BatchAxisContext* ctx) {
  CHECK_OR_RETURN(*ctx->BatchAxis4ArgNameAndIndex("index", 0)
                  == *ctx->BatchAxis4ArgNameAndIndex("input", 0));
  *ctx->BatchAxis4ArgNameAndIndex("output", 0) = *ctx->BatchAxis4ArgNameAndIndex("input", 0);
  return Maybe<void>::Ok();
}

Maybe<void> SetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& index_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("index", 0);
  int64_t index_num_axes = index_tensor.shape().NumAxes();
  const int32_t dim = ctx->Attr<int32_t>("dim");

  FOR_RANGE(int64_t, i, 0, index_num_axes) {
    if (i != dim) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("index", 0), i)
          .Split(user_op::OpArg("input", 0), i)
          .Split(user_op::OpArg("output", 0), i)
          .Split(user_op::OpArg("like", 0), i)
          .Build();
    }
  }

  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("input", 0))
      .Broadcast(user_op::OpArg("index", 0))
      .PartialSum(user_op::OpArg("output", 0))
      .PartialSum(user_op::OpArg("like", 0))
      .Build();
  return Maybe<void>::Ok();
}
}  // namespace

#define REGISTER_SCATTER_LIKE_OP(optypename)   \
  REGISTER_USER_OP(optypename)                 \
      .Input("like")                           \
      .Input("input")                          \
      .Input("index")                          \
      .Output("output")                        \
      .Attr<int32_t>("dim")                    \
      .SetTensorDescInferFn(InferTensorDesc)   \
      .SetInputArgModifyFn(InputArgModifierFn) \
      .SetBatchAxisInferFn(InferBatchAxis)     \
      .SetGetSbpFn(SetSbp)

#define REGISTER_SCATTER_INPLACE_OP(optypename)       \
  REGISTER_USER_OP(optypename)                        \
      .Input("like")                                  \
      .Input("input")                                 \
      .Input("index")                                 \
      .Output("output")                               \
      .Attr<int32_t>("dim")                           \
      .SetTensorDescInferFn(InferTensorDesc)          \
      .SetInputArgModifyFn(InplaceInputArgModifierFn) \
      .SetBatchAxisInferFn(InferBatchAxis)            \
      .SetGetSbpFn(SetSbp)

REGISTER_USER_OP_GRAD("dim_scatter_add_like")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
      const auto op_grad_name = ctx->FwOp().op_name() + "_grad";

      ctx->DefineOp(op_grad_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder
            .OpTypeName("dim_gather")  // dim_gather(grad, dim, index) -> output
            .InputBind("index", ctx->FwOp().input("index", 0))  // gather.index <- scatter.index
            .InputBind("input",
                       ctx->FwOp().output_grad("output", 0))  // gather.input <- grad of scatter.out
            .Output("output")
            .Attr("dim", ctx->FwOp().attr<int32_t>("dim"))
            .Build();
      });

      ctx->FwOp().InputGradBind(user_op::OpArg("input", 0),
                                [&ctx, &op_grad_name]() -> const std::string& {
                                  return ctx->GetOp(op_grad_name).output("output", 0);
                                });
    });

REGISTER_SCATTER_LIKE_OP("dim_scatter_add_like");
REGISTER_SCATTER_LIKE_OP("dim_scatter_update_like");
REGISTER_SCATTER_INPLACE_OP("dim_scatter_add");

}  // namespace user_op
}  // namespace oneflow