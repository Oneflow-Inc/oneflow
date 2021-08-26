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
#include "oneflow/user/ops/reshape_user_op_util.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

Maybe<void> GetSbpFn(user_op::SbpContext* ctx) {
  const auto& in_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape();
  const Shape& shape = ctx->Attr<Shape>("shape");
  const auto& outshape = JUST(ReshapeUserOpUtil::GetLogicalOutBlobShape(in_shape, shape));
  user_op::UserOpSbpSignatureBuilder builder = ctx->NewBuilder();
  return ReshapeUserOpUtil::GetReshapeUserOpSbpSignatures(
      in_shape, *outshape, {{"in", 0}}, {{"out", 0}}, ctx->parallel_num(), &builder);
}

Maybe<void> InferNdSbpFn(user_op::InferNdSbpFnContext* ctx) {
  const Shape& in_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape();
  const Shape& shape = ctx->user_op_conf().attr<Shape>("shape");
  const auto& out_shape = JUST(ReshapeUserOpUtil::GetLogicalOutBlobShape(in_shape, shape));
  return ReshapeUserOpUtil::InferNdSbp(ctx, in_shape, *out_shape);
}

Maybe<void> LogicalTensorDescInferFn(user_op::InferContext* ctx) {
  const Shape& shape = ctx->Attr<Shape>("shape");
  const user_op::TensorDesc& in_tensor_desc = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor_desc = ctx->OutputTensorDesc("out", 0);
  const Shape& in_shape = in_tensor_desc.shape();
  Shape* out_shape = out_tensor_desc->mut_shape();
  CHECK_OR_RETURN(in_tensor_desc.is_dynamic() == false);
  *out_tensor_desc = in_tensor_desc;
  CHECK_GE_OR_RETURN(shape.NumAxes(), 1);
  DimVector dim_vec = {shape.dim_vec().begin(), shape.dim_vec().end()};
  FOR_RANGE(int32_t, i, 0, dim_vec.size()) { CHECK_GE_OR_RETURN(dim_vec.at(i), 0); }
  *out_shape = Shape(dim_vec);
  CHECK_EQ_OR_RETURN(out_shape->elem_cnt(), in_shape.elem_cnt());
  return Maybe<void>::Ok();
}

Maybe<void> TensorDescInferFn(user_op::InferContext* ctx) {
  const Shape& shape = ctx->Attr<Shape>("shape");
  CHECK_GE_OR_RETURN(shape.NumAxes(), 1);
  FOR_RANGE(int32_t, i, 0, shape.NumAxes()) { CHECK_GT_OR_RETURN(shape.At(i), 0); }
  const user_op::TensorDesc& in_tensor_desc = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor_desc = ctx->OutputTensorDesc("out", 0);
  const Shape& in_shape = in_tensor_desc.shape();
  Shape* out_shape = out_tensor_desc->mut_shape();
  CHECK_OR_RETURN(in_tensor_desc.is_dynamic() == false);
  *out_tensor_desc->mut_shape() = in_tensor_desc.shape();
  *out_tensor_desc->mut_is_dynamic() = in_tensor_desc.is_dynamic();
  const auto& nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  *out_shape = *JUST(GetPhysicalShape(shape, nd_sbp, ctx->parallel_desc(), ctx->parallel_ctx()));
  CHECK_EQ_OR_RETURN(out_shape->elem_cnt(), in_shape.elem_cnt());
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP("reshape")
    .Input("in")
    .Output("out")
    .Attr<Shape>("shape")
    .SetLogicalTensorDescInferFn(LogicalTensorDescInferFn)
    .SetPhysicalTensorDescInferFn(TensorDescInferFn)
    .SetGetSbpFn(GetSbpFn)
    .SetNdSbpInferFn(InferNdSbpFn)
    .SetDataTypeInferFn(InferDataType);

REGISTER_USER_OP_GRAD("reshape").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                           user_op::AddOpFn AddOp) -> Maybe<void> {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    const auto& in_desc = op.TensorDesc4ArgNameAndIndex("in", 0);
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    if (in_desc.is_dynamic()) {
      user_op::UserOpConfWrapper reshape_grad_op =
          builder.Op("reshape_like")
              .Input("in", op.GetGradTensorWithOpOutput("out", 0))
              .Input("like", op.input("in", 0))
              .Output("out")
              .Build();
      op.BindGradTensorWithOpInput(reshape_grad_op.output("out", 0), "in", 0);
      AddOp(reshape_grad_op);
    } else {
      user_op::UserOpConfWrapper reshape_grad_op =
          builder.Op("reshape")
              .Input("in", op.GetGradTensorWithOpOutput("out", 0))
              .Attr("shape", in_desc.shape())
              .Output("out")
              .Build();
      op.BindGradTensorWithOpInput(reshape_grad_op.output("out", 0), "in", 0);
      AddOp(reshape_grad_op);
    }
  }
  return Maybe<void>::Ok();
});

}  // namespace
}  // namespace oneflow
