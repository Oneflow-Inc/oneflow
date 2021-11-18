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

int64_t ShiftNegativeAxisIfNeed(const Shape& shape, int64_t axis) {
  const int64_t shifted = axis < 0 ? axis + shape.NumAxes() : axis;
  CHECK_GE(shifted, 0);
  CHECK_LT(shifted, shape.NumAxes());
  return shifted;
}

Shape InferMeanShape(const Shape& x_shape, const int64_t begin_norm_axis) {
  DimVector mean_dim_vec;
  mean_dim_vec.insert(mean_dim_vec.end(), x_shape.dim_vec().cbegin(),
                      x_shape.dim_vec().cbegin() + begin_norm_axis);
  const Shape mean_shape(mean_dim_vec);
  return mean_shape;
}

Shape InferGammaShape(const Shape& x_shape, const int64_t begin_norm_axis) {
  DimVector gamma_dim_vec;
  gamma_dim_vec.insert(gamma_dim_vec.end(), x_shape.dim_vec().cbegin() + begin_norm_axis,
                       x_shape.dim_vec().cend());
  const Shape gamma_shape(gamma_dim_vec);
  return gamma_shape;
}

oneflow::DataType InferMeanDataType(const DataType x_data_type) {
  return x_data_type == DataType::kFloat16 ? DataType::kFloat : x_data_type;
}

}  // namespace

REGISTER_USER_OP("layer_norm")
    .Input("x")
    .OptionalInput("gamma")
    .OptionalInput("beta")
    .Output("y")
    .Output("mean")
    .Output("inv_variance")
    .Attr<int64_t>("begin_norm_axis")
    .Attr<double>("epsilon")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
      user_op::TensorDesc* y = ctx->OutputTensorDesc("y", 0);
      user_op::TensorDesc* mean = ctx->OutputTensorDesc("mean", 0);
      user_op::TensorDesc* inv_variance = ctx->OutputTensorDesc("inv_variance", 0);
      const int64_t begin_norm_axis =
          ShiftNegativeAxisIfNeed(x.shape(), ctx->Attr<int64_t>("begin_norm_axis"));
      const Shape& gamma_shape = InferGammaShape(x.shape(), begin_norm_axis);
      if (ctx->has_input("gamma", 0)) {
        const user_op::TensorDesc& gamma = ctx->InputTensorDesc("gamma", 0);
        CHECK_EQ_OR_RETURN(gamma.shape(), gamma_shape);
      }
      if (ctx->has_input("beta", 0)) {
        const user_op::TensorDesc& beta = ctx->InputTensorDesc("beta", 0);
        CHECK_EQ_OR_RETURN(beta.shape(), gamma_shape);
      }
      *y->mut_shape() = x.shape();
      *y->mut_is_dynamic() = x.is_dynamic();
      *mean->mut_shape() = InferMeanShape(x.shape(), begin_norm_axis);
      *inv_variance = *mean;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
      int64_t begin_norm_axis =
          ShiftNegativeAxisIfNeed(x_shape, ctx->Attr<int64_t>("begin_norm_axis"));
      for (int i = 0; i < begin_norm_axis; ++i) {
        ctx->NewBuilder()
            .Split(ctx->inputs(), i)
            .Split(ctx->outputs(), i)
            .Broadcast(user_op::OpArg("gamma", 0))
            .Broadcast(user_op::OpArg("beta", 0))
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
      user_op::TensorDesc* y = ctx->OutputTensorDesc("y", 0);
      user_op::TensorDesc* mean = ctx->OutputTensorDesc("mean", 0);
      user_op::TensorDesc* inv_variance = ctx->OutputTensorDesc("inv_variance", 0);
      if (ctx->has_input("gamma", 0)) {
        const user_op::TensorDesc& gamma = ctx->InputTensorDesc("gamma", 0);
        CHECK_EQ_OR_RETURN(gamma.data_type(), x.data_type());
      }
      if (ctx->has_input("beta", 0)) {
        const user_op::TensorDesc& beta = ctx->InputTensorDesc("beta", 0);
        CHECK_EQ_OR_RETURN(beta.data_type(), x.data_type());
      }
      *y->mut_data_type() = x.data_type();
      *mean->mut_data_type() = InferMeanDataType(x.data_type());
      *inv_variance->mut_data_type() = mean->data_type();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("layer_norm_grad")
    .Input("dy")
    .Input("x")
    .Input("mean")
    .Input("inv_variance")
    .OptionalInput("gamma")
    .OptionalInput("_add_to_output")
    .OptionalOutput("dx")
    .OptionalOutput("gamma_diff")
    .OptionalOutput("beta_diff")
    .Attr<int64_t>("begin_norm_axis")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
      const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
      const user_op::TensorDesc& mean = ctx->InputTensorDesc("mean", 0);
      const user_op::TensorDesc& inv_variance = ctx->InputTensorDesc("inv_variance", 0);
      CHECK_EQ_OR_RETURN(dy.shape(), x.shape());
      const int64_t begin_norm_axis = ctx->Attr<int64_t>("begin_norm_axis");
      CHECK_GT_OR_RETURN(begin_norm_axis, 0);
      const Shape& mean_shape = InferMeanShape(x.shape(), begin_norm_axis);
      CHECK_EQ_OR_RETURN(mean.shape(), mean_shape);
      CHECK_EQ_OR_RETURN(inv_variance.shape(), mean_shape);
      const Shape& gamma_shape = InferGammaShape(x.shape(), begin_norm_axis);
      if (ctx->has_input("gamma", 0)) {
        const user_op::TensorDesc& gamma = ctx->InputTensorDesc("gamma", 0);
        CHECK_EQ_OR_RETURN(gamma.shape(), gamma_shape);
      }
      if (ctx->has_input("_add_to_output", 0)) {
        const auto& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
        CHECK_EQ_OR_RETURN(add_to_output.shape(), dy.shape());
      }
      if (ctx->has_output("dx", 0)) {
        user_op::TensorDesc* dx = ctx->OutputTensorDesc("dx", 0);
        *dx->mut_shape() = dy.shape();
        *dx->mut_is_dynamic() = dy.is_dynamic();
      }
      if (ctx->has_output("gamma_diff", 0)) {
        user_op::TensorDesc* gamma_diff = ctx->OutputTensorDesc("gamma_diff", 0);
        *gamma_diff->mut_shape() = gamma_shape;
      }
      if (ctx->has_output("beta_diff", 0)) {
        user_op::TensorDesc* beta_diff = ctx->OutputTensorDesc("beta_diff", 0);
        *beta_diff->mut_shape() = gamma_shape;
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
      int64_t begin_norm_axis =
          ShiftNegativeAxisIfNeed(x_shape, ctx->Attr<int64_t>("begin_norm_axis"));
      for (int i = 0; i < begin_norm_axis; ++i) {
        user_op::UserOpSbpSignatureBuilder builder =
            ctx->NewBuilder()
                .Split(user_op::OpArg("dy", 0), i)
                .Split(user_op::OpArg("x", 0), i)
                .Split(user_op::OpArg("mean", 0), i)
                .Split(user_op::OpArg("inv_variance", 0), i);
        if (ctx->user_op_conf().has_input("gamma", 0)) {
          builder.Broadcast(user_op::OpArg("gamma", 0));
        }
        if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
          builder.Split(user_op::OpArg("_add_to_output", 0), i);
        }
        if (ctx->user_op_conf().has_output("dx", 0)) { builder.Split(user_op::OpArg("dx", 0), i); }
        if (ctx->user_op_conf().has_output("gamma_diff", 0)) {
          builder.PartialSum(user_op::OpArg("gamma_diff", 0));
        }
        if (ctx->user_op_conf().has_output("beta_diff", 0)) {
          builder.PartialSum(user_op::OpArg("beta_diff", 0));
        }
        builder.Build();
      }
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
      const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
      CHECK_EQ_OR_RETURN(dy.data_type(), x.data_type());
      const user_op::TensorDesc& mean = ctx->InputTensorDesc("mean", 0);
      const user_op::TensorDesc& inv_variance = ctx->InputTensorDesc("inv_variance", 0);
      const DataType& mean_data_type = InferMeanDataType(x.data_type());
      CHECK_EQ_OR_RETURN(mean.data_type(), mean_data_type);
      CHECK_EQ_OR_RETURN(inv_variance.data_type(), mean_data_type);
      if (ctx->has_input("gamma", 0)) {
        const user_op::TensorDesc& gamma = ctx->InputTensorDesc("gamma", 0);
        CHECK_EQ_OR_RETURN(gamma.data_type(), dy.data_type());
      }
      if (ctx->has_input("_add_to_output", 0)) {
        const auto& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
        CHECK_EQ_OR_RETURN(add_to_output.data_type(), dy.data_type());
      }
      if (ctx->has_output("beta_diff", 0)) {
        user_op::TensorDesc* beta_diff = ctx->OutputTensorDesc("beta_diff", 0);
        *beta_diff->mut_data_type() = dy.data_type();
      }
      if (ctx->has_output("gamma_diff", 0)) {
        user_op::TensorDesc* gamma_diff = ctx->OutputTensorDesc("gamma_diff", 0);
        *gamma_diff->mut_data_type() = dy.data_type();
      }
      if (ctx->has_output("dx", 0)) {
        user_op::TensorDesc* dx = ctx->OutputTensorDesc("dx", 0);
        *dx->mut_data_type() = dy.data_type();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("layer_norm")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      const bool has_x_diff = op.NeedGenGradTensor4OpInput("x", 0);
      const bool has_gamma_diff =
          op.user_op_conf().has_input("gamma", 0) && op.NeedGenGradTensor4OpInput("gamma", 0);
      const bool has_beta_diff =
          op.user_op_conf().has_input("beta", 0) && op.NeedGenGradTensor4OpInput("beta", 0);
      const int64_t begin_norm_axis = ShiftNegativeAxisIfNeed(
          op.TensorDesc4ArgNameAndIndex("x", 0).shape(), op.attr<int64_t>("begin_norm_axis"));
      std::string dy = op.GetGradTensorWithOpOutput("y", 0);
      if (has_x_diff || has_gamma_diff || has_beta_diff) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        auto grad_op_builder = builder.Op("layer_norm_grad")
                                   .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                   .Input("x", op.input("x", 0))
                                   .Input("mean", op.output("mean", 0))
                                   .Input("inv_variance", op.output("inv_variance", 0))
                                   .Attr("begin_norm_axis", begin_norm_axis);
        if (op.user_op_conf().has_input("gamma", 0)) {
          grad_op_builder.Input("gamma", op.input("gamma", 0));
        }
        if (has_x_diff) { grad_op_builder.Output("dx"); }
        if (has_gamma_diff) { grad_op_builder.Output("gamma_diff"); }
        if (has_beta_diff) { grad_op_builder.Output("beta_diff"); }
        auto grad_op = grad_op_builder.Build();
        if (has_x_diff) { op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0); }
        if (has_gamma_diff) {
          op.BindGradTensorWithOpInput(grad_op.output("gamma_diff", 0), "gamma", 0);
        }
        if (has_beta_diff) {
          op.BindGradTensorWithOpInput(grad_op.output("beta_diff", 0), "beta", 0);
        }
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
