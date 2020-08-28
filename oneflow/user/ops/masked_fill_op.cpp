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

bool IsScalarTensor(const user_op::TensorDesc* tensor) {
  return tensor->shape().NumAxes() == 1 && tensor->shape().At(0) == 1;
}
}  // namespace

REGISTER_USER_OP("masked_fill")
    .Input("x")
    .Input("mask")
    .Attr("value", UserOpAttrType::kAtFloat)
    .Output("y")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      const user_op::TensorDesc* mask = ctx->TensorDesc4ArgNameAndIndex("mask", 0);
      user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      size_t output_num_axes = std::max(x->shape().NumAxes(), mask->shape().NumAxes());
      if (IsScalarTensor(mask)) {
        *y = *x;
      } else if (IsScalarTensor(x)) {
        *y = *mask;
      } else {
        const auto& x_shape = CreateLeftExtendedShape(ShapeView(x->shape()), output_num_axes);
        const auto& mask_shape = CreateLeftExtendedShape(ShapeView(mask->shape()), output_num_axes);
        *y = *x;
        Shape out_shape(x_shape);
        FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
          CHECK_OR_RETURN(x_shape.At(i) == 1 || mask_shape.At(i) == 1
                          || x_shape.At(i) == mask_shape.At(i))
              << "op: " << ctx->user_op_conf().op_name()
              << ", type: " << ctx->user_op_conf().op_type_name() << ", i: " << i
              << ", x_shape: " << x_shape << ", mask_shape: " << mask_shape;
          out_shape.Set(i, std::max(x_shape.At(i), mask_shape.At(i)));
        }
        *y->mut_shape() = out_shape;
      }
      y->set_is_dynamic(x->is_dynamic());
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* cond_arg_modifier = GetInputArgModifierFn("mask", 0);
      cond_arg_modifier->set_requires_grad(false);
    })
    .SetGetSbpFn(user_op::GetSbpFnUtil::SplitForEachAxis);

REGISTER_USER_OP("masked_fill_grad")
    .Input("dy")
    .Input("x")
    .Input("mask")
    .Output("dx")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* dx = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
      *dx = *x;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn(user_op::GetSbpFnUtil::SplitForEachAxis);

REGISTER_USER_OP_GRAD("masked_fill")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("masked_fill_grad")
                                                 .Input("x", op.input("x", 0))
                                                 .Input("mask", op.input("mask", 0))
                                                 .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                                 .Output("dx")
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
