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

namespace {

int64_t ShiftNegativeAxisIfNeed(const Shape& shape, int64_t axis) {
  const int64_t shifted = axis < 0 ? axis + shape.NumAxes() : axis;
  CHECK_GE(shifted, 0);
  CHECK_LT(shifted, shape.NumAxes());
  return shifted;
}

Shape InferBnParamShape(const Shape& x_shape, const int64_t begin_norm_axis) {
  DimVector bn_param_shape_dim_vec;
  bn_param_shape_dim_vec.insert(bn_param_shape_dim_vec.end(), x_shape.dim_vec().cbegin(),
                                x_shape.dim_vec().cbegin() + begin_norm_axis);
  const Shape bn_param_shape(bn_param_shape_dim_vec);
  return bn_param_shape;
}

oneflow::DataType InferBnParamDataType(const DataType x_data_type) {
  return x_data_type == DataType::kFloat16 ? DataType::kFloat : x_data_type;
}

}  // namespace

/* static */ Maybe<void> LayerNormOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  user_op::TensorDesc* y = ctx->OutputTensorDesc("y", 0);
  user_op::TensorDesc* mean = ctx->OutputTensorDesc("mean", 0);
  user_op::TensorDesc* inv_variance = ctx->OutputTensorDesc("inv_variance", 0);
  const bool center = ctx->Attr<bool>("center");
  const bool scale = ctx->Attr<bool>("scale");
  const int64_t begin_params_axis =
      ShiftNegativeAxisIfNeed(x.shape(), ctx->Attr<int64_t>("begin_params_axis"));
  *y->mut_shape() = x.shape();
  *y->mut_is_dynamic() = x.is_dynamic();
  DimVector param_shape_dim_vec;
  param_shape_dim_vec.insert(param_shape_dim_vec.end(),
                             x.shape().dim_vec().cbegin() + begin_params_axis,
                             x.shape().dim_vec().cend());
  const Shape param_shape(param_shape_dim_vec);
  if (center) {
    const user_op::TensorDesc& beta = ctx->InputTensorDesc("beta", 0);
    CHECK_EQ_OR_RETURN(beta.shape(), param_shape);
  }
  if (scale) {
    const user_op::TensorDesc& gamma = ctx->InputTensorDesc("gamma", 0);
    CHECK_EQ_OR_RETURN(gamma.shape(), param_shape);
  }
  const int64_t begin_norm_axis =
      ShiftNegativeAxisIfNeed(x.shape(), ctx->Attr<int64_t>("begin_norm_axis"));
  if (begin_norm_axis != begin_params_axis) {
    return Error::RuntimeError() << "begin_norm_axis must equal to begin_params_axis, but got "
                                 << begin_norm_axis << " vs " << begin_params_axis;
  }
  *mean->mut_shape() = InferBnParamShape(x.shape(), begin_norm_axis);
  *inv_variance = *mean;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> LayerNormOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> LayerNormOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  int64_t begin_norm_axis = ShiftNegativeAxisIfNeed(x_shape, ctx->Attr<int64_t>("begin_norm_axis"));
  int64_t begin_params_axis =
      ShiftNegativeAxisIfNeed(x_shape, ctx->Attr<int64_t>("begin_params_axis"));
  for (int i = 0; i < std::min(begin_norm_axis, begin_params_axis); ++i) {
    ctx->NewBuilder()
        .Split(ctx->inputs(), i)
        .Split(ctx->outputs(), i)
        .Broadcast(user_op::OpArg("gamma", 0))
        .Broadcast(user_op::OpArg("beta", 0))
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> LayerNormOp::InferDataType(user_op::InferContext* ctx) {
  const bool center = ctx->Attr<bool>("center");
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  user_op::TensorDesc* y = ctx->OutputTensorDesc("y", 0);
  *y->mut_data_type() = x.data_type();
  if (center) {
    const user_op::TensorDesc& beta = ctx->InputTensorDesc("beta", 0);
    CHECK_EQ_OR_RETURN(beta.data_type(), x.data_type());
  }
  const bool scale = ctx->Attr<bool>("scale");
  if (scale) {
    const user_op::TensorDesc& gamma = ctx->InputTensorDesc("gamma", 0);
    CHECK_EQ_OR_RETURN(gamma.data_type(), x.data_type());
  }
  user_op::TensorDesc* mean = ctx->OutputTensorDesc("mean", 0);
  user_op::TensorDesc* inv_variance = ctx->OutputTensorDesc("inv_variance", 0);
  *mean->mut_data_type() = InferBnParamDataType(x.data_type());
  *inv_variance->mut_data_type() = mean->data_type();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> LayerNormGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& mean = ctx->InputTensorDesc("mean", 0);
  const user_op::TensorDesc& inv_variance = ctx->InputTensorDesc("inv_variance", 0);
  user_op::TensorDesc* dx = ctx->OutputTensorDesc("dx", 0);
  CHECK_EQ_OR_RETURN(dy.shape(), x.shape());
  const int64_t begin_norm_axis = ctx->Attr<int64_t>("begin_norm_axis");
  CHECK_GT_OR_RETURN(begin_norm_axis, 0);
  const Shape& bn_param_shape = InferBnParamShape(x.shape(), begin_norm_axis);
  CHECK_EQ_OR_RETURN(mean.shape(), bn_param_shape);
  CHECK_EQ_OR_RETURN(inv_variance.shape(), bn_param_shape);
  *dx->mut_shape() = dy.shape();
  *dx->mut_is_dynamic() = dy.is_dynamic();
  if (ctx->has_input("_add_to_output", 0)) {
    const auto& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
    CHECK_EQ_OR_RETURN(add_to_output.shape(), dx->shape());
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> LayerNormGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> LayerNormGradOp::GetSbp(user_op::SbpContext* ctx) {
  std::vector<user_op::OpArg> broadcast_args;
  if (ctx->user_op_conf().has_input("gamma", 0)) {
    broadcast_args.emplace_back(user_op::OpArg("gamma", 0));
  }
  int64_t begin_norm_axis = ctx->Attr<int64_t>("begin_norm_axis");
  for (int i = 0; i < begin_norm_axis; ++i) {
    ctx->NewBuilder()
        .Split(ctx->inputs(), i)
        .Split(ctx->outputs(), i)
        .Broadcast(broadcast_args)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> LayerNormGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  CHECK_EQ_OR_RETURN(dy.data_type(), x.data_type());
  const user_op::TensorDesc& mean = ctx->InputTensorDesc("mean", 0);
  const user_op::TensorDesc& inv_variance = ctx->InputTensorDesc("inv_variance", 0);
  DataType bn_param_data_type = InferBnParamDataType(x.data_type());
  CHECK_EQ_OR_RETURN(mean.data_type(), bn_param_data_type);
  CHECK_EQ_OR_RETURN(inv_variance.data_type(), bn_param_data_type);
  user_op::TensorDesc* dx = ctx->OutputTensorDesc("dx", 0);
  *dx->mut_data_type() = dy.data_type();
  if (ctx->has_input("_add_to_output", 0)) {
    const auto& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
    CHECK_EQ_OR_RETURN(add_to_output.data_type(), dx->data_type());
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> LayerNormParamGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  // TODO: tsai: replace lambda with user op if
  auto has_tensor = [ctx](const std::string& bn) -> bool {
    bool ret = false;
    for (const auto& t : ctx->inputs()) {
      if (bn == t.first) { return true; }
    }
    for (const auto& t : ctx->outputs()) {
      if (bn == t.first) { return true; }
    }
    return ret;
  };
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  const int64_t begin_params_axis = ctx->Attr<int64_t>("begin_params_axis");
  const bool has_beta_diff = has_tensor("beta_diff");
  const bool has_gamma_diff = has_tensor("gamma_diff");
  CHECK_GE_OR_RETURN(begin_params_axis, 1);
  CHECK_LT_OR_RETURN(begin_params_axis, dy.shape().NumAxes());
  DimVector param_shape_dim_vec;
  param_shape_dim_vec.insert(param_shape_dim_vec.end(),
                             dy.shape().dim_vec().cbegin() + begin_params_axis,
                             dy.shape().dim_vec().cend());
  const Shape param_shape(param_shape_dim_vec);
  if (has_beta_diff) {
    user_op::TensorDesc* beta_diff = ctx->OutputTensorDesc("beta_diff", 0);
    *beta_diff->mut_shape() = param_shape;
  }
  if (has_gamma_diff) {
    user_op::TensorDesc* gamma_diff = ctx->OutputTensorDesc("gamma_diff", 0);
    *gamma_diff->mut_shape() = param_shape;
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> LayerNormParamGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> LayerNormParamGradOp::GetSbp(user_op::SbpContext* ctx) {
  int64_t begin_params_axis = ctx->Attr<int64_t>("begin_params_axis");
  for (int i = 0; i < begin_params_axis; ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i).PartialSum(ctx->outputs()).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> LayerNormParamGradOp::InferDataType(user_op::InferContext* ctx) {
  auto has_tensor = [ctx](const std::string& bn) -> bool {
    bool ret = false;
    for (auto& t : ctx->inputs()) {
      if (bn == t.first) { return true; }
    }
    for (auto& t : ctx->outputs()) {
      if (bn == t.first) { return true; }
    }
    return ret;
  };
  const bool has_beta_diff = has_tensor("beta_diff");
  const bool has_gamma_diff = has_tensor("gamma_diff");
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  if (has_beta_diff) {
    user_op::TensorDesc* beta_diff = ctx->OutputTensorDesc("beta_diff", 0);
    *beta_diff->mut_data_type() = dy.data_type();
  }
  if (has_gamma_diff) {
    user_op::TensorDesc* gamma_diff = ctx->OutputTensorDesc("gamma_diff", 0);
    *gamma_diff->mut_data_type() = dy.data_type();
  }
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("layer_norm")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      const bool center = op.attr<bool>("center");
      const bool scale = op.attr<bool>("scale");
      const bool has_beta = center;
      const bool has_gamma = scale;
      const bool has_beta_diff = has_beta && op.NeedGenGradTensor4OpInput("beta", 0);
      const bool has_gamma_diff = has_gamma && op.NeedGenGradTensor4OpInput("gamma", 0);
      const Shape& x_shape = op.TensorDesc4ArgNameAndIndex("x", 0).shape();
      const int64_t begin_norm_axis =
          ShiftNegativeAxisIfNeed(x_shape, op.attr<int64_t>("begin_norm_axis"));
      const int64_t begin_params_axis =
          ShiftNegativeAxisIfNeed(x_shape, op.attr<int64_t>("begin_params_axis"));
      if (has_beta_diff || has_gamma_diff) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_param_grad");
        auto grad_op_builder = builder.Op("layer_norm_param_grad")
                                   .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                   .Input("x", op.input("x", 0))
                                   .Input("mean", op.output("mean", 0))
                                   .Input("inv_variance", op.output("inv_variance", 0))
                                   .Attr("begin_params_axis", begin_params_axis);
        if (has_beta_diff) { grad_op_builder.Output("beta_diff"); }
        if (has_gamma_diff) { grad_op_builder.Output("gamma_diff"); }
        auto grad_op = grad_op_builder.Build();
        if (has_beta_diff) {
          op.BindGradTensorWithOpInput(grad_op.output("beta_diff", 0), "beta", 0);
        }
        if (has_gamma_diff) {
          op.BindGradTensorWithOpInput(grad_op.output("gamma_diff", 0), "gamma", 0);
        }
        AddOp(grad_op);
      }
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        builder.Op("layer_norm_grad")
            .Input("x", op.input("x", 0))
            .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
            .Input("mean", op.output("mean", 0))
            .Input("inv_variance", op.output("inv_variance", 0))
            .Output("dx")
            .Attr("begin_norm_axis", begin_norm_axis)
            .Attr("epsilon", op.attr<double>("epsilon"));
        if (op.user_op_conf().has_input("gamma", 0)) {
          builder.Input("gamma", op.input("gamma", 0));
        }
        user_op::UserOpConfWrapper grad_op = builder.Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
