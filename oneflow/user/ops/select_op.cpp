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

/* static */ Maybe<void> SelectOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input = ctx->InputTensorDesc("input", 0);
  user_op::TensorDesc* output = ctx->OutputTensorDesc("output", 0);
  const int32_t dim = ctx->Attr<int32_t>("dim");
  const int32_t index = ctx->Attr<int32_t>("index");
  
  DimVector dim_vec;
  dim_vec.insert(dim_vec.end(), input.shape().dim_vec().cbegin(), input.shape().dim_vec().cbegin() + dim);
  dim_vec.insert(dim_vec.end(), input.shape().dim_vec().cbegin() + dim + 1, input.shape().dim_vec().cend());
  *output->mut_shape() = Shape(dim_vec);
  
  output->set_is_dynamic(input.is_dynamic());

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SelectOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> SelectOp::GetSbp(user_op::SbpContext* ctx) {
  const int32_t input_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0).shape().NumAxes();
  const int32_t dim = ctx->Attr<int32_t>("dim");
  
  FOR_RANGE(int32_t, i, 0, input_num_axes) {
      if(i<dim){
        ctx->NewBuilder()
        .Split(user_op::OpArg("input", 0), i)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
      }else if(i>dim){
        ctx->NewBuilder()
        .Split(user_op::OpArg("input", 0), i)
        .Split(user_op::OpArg("out", 0), i-1)
        .Build();
      }
  }

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SelectOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("output", 0) = ctx->InputDType("input", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SelectGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("dx", 0) = ctx->InputShape("input", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SelectGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> SelectGradOp::GetSbp(user_op::SbpContext* ctx) {
  const int32_t dx_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("dx", 0).shape().NumAxes(  );
  const int32_t dim = ctx->Attr<int32_t>("dim");
  
  FOR_RANGE(int32_t, i, 0, dx_num_axes) {
      if(i<dim){
        ctx->NewBuilder()
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
      }else if(i>dim){
        ctx->NewBuilder()
        .Split(user_op::OpArg("dy", 0), i-1)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
      }
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SelectGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("select")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) -> Maybe<void> {
      const auto grad_op_name = ctx->FwOp().op_name() + "_grad";
      ctx->DefineOp(grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("select_grad")
            .InputBind("input", ctx->FwOp().input("input", 0))
            .InputBind("dy", ctx->FwOp().output_grad("output", 0))
            .Attr<int32_t>("dim", ctx->FwOp().attr<int32_t>("dim"))
            .Attr<int32_t>("index", ctx->FwOp().attr<int32_t>("index"))
            .Output("dx")
            .Build();
      });

      ctx->FwOp().InputGradBind(user_op::OpArg("input", 0),
                                [&ctx, &grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(grad_op_name).output("dx", 0);
                                });
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
