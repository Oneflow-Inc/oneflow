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
#include "oneflow/user/ops/nn_util.h"
#include "pad2d_seq.h"

namespace oneflow {

namespace {

Maybe<void> GetOpSbpSignature(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    if (padding[i] == 0) {
      ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetOpGradSbpSignature(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& dy_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0);
  const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");
  FOR_RANGE(int64_t, i, 0, dy_tensor.shape().NumAxes()) {
    if (padding[i] == 0) {
      ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

#define REGISTER_PAD_2D_OP_AND_GRAD(pad_2d_type)                                             \
  REGISTER_USER_OP(pad_2d_type)                                                              \
      .Input("x")                                                                            \
      .Output("y")                                                                           \
      .Attr<std::vector<int64_t>>("padding")                                                 \
      .Attr<double>("floating_value")                                                        \
      .Attr<int64_t>("integral_value")                                                       \
      .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {                  \
        Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);                                 \
        const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");                    \
        CHECK_EQ_OR_RETURN(padding.size(), x_shape->NumAxes());                              \
        const int64_t n_idx = 0;                                                             \
        const int64_t c_idx = 1;                                                             \
        const int64_t h_idx = 2;                                                             \
        const int64_t w_idx = 3;                                                             \
        CHECK_LT_OR_RETURN(padding[0], x_shape->At(w_idx));                                  \
        CHECK_LT_OR_RETURN(padding[1], x_shape->At(w_idx));                                  \
        CHECK_LT_OR_RETURN(padding[2], x_shape->At(h_idx));                                  \
        CHECK_LT_OR_RETURN(padding[3], x_shape->At(h_idx));                                  \
                                                                                             \
        DimVector y_dim_vec(x_shape->NumAxes());                                             \
        const int64_t h_x = x_shape->At(h_idx);                                              \
        const int64_t w_x = x_shape->At(w_idx);                                              \
                                                                                             \
        y_dim_vec[n_idx] = x_shape->At(n_idx);                                               \
        y_dim_vec[c_idx] = x_shape->At(c_idx);                                               \
        y_dim_vec[h_idx] = h_x + padding[2] + padding[3];                                    \
        y_dim_vec[w_idx] = w_x + padding[0] + padding[1];                                    \
                                                                                             \
        *ctx->Shape4ArgNameAndIndex("y", 0) = Shape(y_dim_vec);                              \
        return Maybe<void>::Ok();                                                            \
      })                                                                                     \
      .SetGetSbpFn(GetOpSbpSignature)                                                        \
      .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,            \
                              const user_op::UserOpConfWrapper&) {                           \
        user_op::InputArgModifier* x_modifier = GetInputArgModifierFn("x", 0);               \
        CHECK_NOTNULL(x_modifier);                                                           \
        x_modifier->set_requires_grad(true);                                                 \
      })                                                                                    \
      .SetInferDataTypeFn([](user_op::InferContext* ctx) -> Maybe<void> {                        \
        *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);           \
        return Maybe<void>::Ok();                                                            \
                        });  \
                                                                                             \
  REGISTER_USER_OP((std::string("") + pad_2d_type + "_grad"))                                \
      .Input("dy")                                                                           \
      .Output("dx")                                                                          \
      .Attr<std::vector<int64_t>>("padding")                                                 \
      .Attr<double>("floating_value")                                                        \
      .Attr<int64_t>("integral_value")                                                       \
      .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {                  \
        Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);                               \
        const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");                    \
        CHECK_EQ_OR_RETURN(padding.size(), dy_shape->NumAxes());                             \
        const int64_t n_idx = 0;                                                             \
        const int64_t c_idx = 1;                                                             \
        const int64_t h_idx = 2;                                                             \
        const int64_t w_idx = 3;                                                             \
                                                                                             \
        DimVector dx_dim_vec(dy_shape->NumAxes());                                           \
        int64_t h_dy, w_dy;                                                                  \
        h_dy = dy_shape->At(h_idx);                                                          \
        w_dy = dy_shape->At(w_idx);                                                          \
                                                                                             \
        dx_dim_vec[n_idx] = dy_shape->At(0);                                                 \
        dx_dim_vec[c_idx] = dy_shape->At(1);                                                 \
        dx_dim_vec[h_idx] = h_dy - padding[2] - padding[3];                                  \
        dx_dim_vec[w_idx] = w_dy - padding[0] - padding[1];                                  \
                                                                                             \
        *ctx->Shape4ArgNameAndIndex("dx", 0) = Shape(dx_dim_vec);                            \
        return Maybe<void>::Ok();                                                            \
      })                                                                                     \
      .SetGetSbpFn(GetOpGradSbpSignature)                                                   \
      .SetInferDataTypeFn([](user_op::InferContext* ctx) -> Maybe<void> {                        \
        *ctx->Dtype4ArgNameAndIndex("dx", 0) = *ctx->Dtype4ArgNameAndIndex("dy", 0);         \
        return Maybe<void>::Ok();                                                            \
                        });  \
                                                                                             \
  REGISTER_USER_OP_GRAD(pad_2d_type)                                                         \
      .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) { \
        if (op.NeedGenGradTensor4OpInput("x", 0)) {                                          \
          user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");                 \
          user_op::UserOpConfWrapper grad_op =                                               \
              builder.Op((std::string("") + pad_2d_type + "_grad"))                          \
                  .Input("dy", op.GetGradTensorWithOpOutput("y", 0))                         \
                  .Output("dx")                                                              \
                  .Attr("padding", op.attr<std::vector<int64_t>>("padding"))                 \
                  .Attr("floating_value", op.attr<double>("floating_value"))                 \
                  .Attr("integral_value", op.attr<int64_t>("integral_value"))                \
                  .Build();                                                                  \
          op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);                     \
          AddOp(grad_op);                                                                    \
        }                                                                                    \
      });

OF_PP_FOR_EACH_TUPLE(REGISTER_PAD_2D_OP_AND_GRAD, PAD_2D_TYPE_SEQ)

}  // namespace oneflow
