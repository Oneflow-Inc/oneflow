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

namespace user_op {
REGISTER_USER_OP("gather_dim")
    .Input("input")
    .Input("index")
    .Output("out")
    .Attr("dim", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("input", 0);
      int64_t input_num_axes = in->shape().NumAxes();
      CHECK_GT_OR_RETURN(input_num_axes, 0);
      CHECK_LE_OR_RETURN(input_num_axes, 8);

      const TensorDesc* index = ctx->TensorDesc4ArgNameAndIndex("index", 0);
      int64_t index_num_axes = index->shape().NumAxes();
      CHECK_GT_OR_RETURN(index_num_axes, 0);
      CHECK_LE_OR_RETURN(index_num_axes, 8);
      CHECK_OR_RETURN(IsIndexDataType(index->data_type()));

      const int64_t dim = ctx->Attr<int64_t>("dim");
      CHECK_GE_OR_RETURN(dim, 0);
      CHECK_EQ_OR_RETURN(input_num_axes, index_num_axes);

      FOR_RANGE(int64_t, i, 0, input_num_axes) {
        if (i == dim) { continue; }
        CHECK_EQ_OR_RETURN(in->shape().At(i), index->shape().At(i));
      }

      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out->mut_shape() = index->shape();
      *out->mut_data_type() = in->data_type();

      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("scatter_dim_add")
    .Input("src")
    .Input("index")
    .Output("out")
    .Attr("dim", UserOpAttrType::kAtInt64)
    .Attr("shape", UserOpAttrType::kAtShape)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("src", 0);
      const TensorDesc* index = ctx->TensorDesc4ArgNameAndIndex("index", 0);
      const Shape& params_shape = ctx->Attr<Shape>("shape");
      int64_t dim = ctx->Attr<int64_t>("dim");

      int64_t src_num_axes = src->shape().NumAxes();
      CHECK_GT_OR_RETURN(src_num_axes, 0);
      CHECK_LE_OR_RETURN(src_num_axes, 8);

      int64_t index_num_axes = index->shape().NumAxes();
      CHECK_EQ_OR_RETURN(src_num_axes, index_num_axes);
      CHECK_LE_OR_RETURN(src_num_axes, params_shape.NumAxes());

      FOR_RANGE(int64_t, i, 0, src_num_axes) {
        CHECK_LE_OR_RETURN(index->shape().At(i), src->shape().At(i));
      }

      FOR_RANGE(int64_t, i, 0, src_num_axes) {
        if (i == dim) { continue; }
        CHECK_LE_OR_RETURN(index->shape().At(i), params_shape.At(i));
      }

      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out->mut_shape() = params_shape;
      *out->mut_data_type() = src->data_type();

      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("gather_dim").SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {

  const auto op_grad_name = ctx->FwOp().op_name() + "_grad";

  ctx->DefineOp(op_grad_name, [&ctx](user_op::BackwardOpBuilder& builder) {
    return builder
        .OpTypeName("scatter_dim_add")  // scatter_dim_add(dim, index, src) -> output
        .InputBind("index", ctx->FwOp().input("index", 0))    // scatter.index <- gather.index
        .InputBind("src", ctx->FwOp().output_grad("out", 0))  // scatter.src <- grad of gather.out
        .Output("out")
        .Attr("dim", ctx->FwOp().attr<int64_t>("dim"))
        .Attr("shape", ctx->FwOp().TensorDesc4ArgNameAndIndex("out", 0).shape())
        .Build();
  });

  ctx->FwOp().InputGradBind(user_op::OpArg("input", 0),
                            [&ctx, &op_grad_name]() -> const std::string& {
                              return ctx->GetOp(op_grad_name).output("out", 0);
                            });
});

}  // namespace user_op

}  // namespace oneflow
