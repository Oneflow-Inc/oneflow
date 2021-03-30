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

namespace oneflow {
namespace {

Maybe<void> TensorDescInfer(user_op::InferContext* ctx) {
  const user_op::TensorDesc* input_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  user_op::TensorDesc* output_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  *output_tensor_desc->mut_shape() = input_tensor_desc->shape();
  *output_tensor_desc->mut_is_dynamic() = input_tensor_desc->is_dynamic();
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignatures(user_op::SbpContext* ctx) {
  const auto& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  for (int i = 0; i < in_tensor.shape().NumAxes(); ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType(user_op::InferContext* ctx) {
  user_op::TensorDesc* output_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  DataType* dtype = output_tensor_desc->mut_data_type();
  *dtype = ctx->Attr<DataType>("dtype");
  return Maybe<void>::Ok();
}

REGISTER_USER_OP("cast")
    .Input("in")
    .Attr<DataType>("dtype")
    .Output("out")
    .SetTensorDescInferFn(TensorDescInfer)
    .SetGetSbpFn(GetSbpSignatures)
    .SetInferDataTypeFn(InferDataType);

REGISTER_USER_OP_GRAD("cast").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                        user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    const DataType& dtype = op.TensorDesc4ArgNameAndIndex("in", 0).data_type();
    user_op::UserOpConfWrapper cast_grad_op =
        builder.Op("cast")
            .Input("in", op.GetGradTensorWithOpOutput("out", 0))
            .Output("out")
            .Attr<DataType>("dtype", dtype)
            .Build();
    op.BindGradTensorWithOpInput(cast_grad_op.output("out", 0), "in", 0);
    AddOp(cast_grad_op);
  }
});

}  // namespace
}  // namespace oneflow
