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
  return (x_data_type == DataType::kFloat16 || x_data_type == DataType::kBFloat16)
             ? DataType::kFloat
             : x_data_type;
}

}  // namespace

/* static */ Maybe<void> LayerNormOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  user_op::TensorDesc* y = ctx->MutOutputTensorDesc("y", 0);
  user_op::TensorDesc* mean = ctx->MutOutputTensorDesc("mean", 0);
  user_op::TensorDesc* inv_variance = ctx->MutOutputTensorDesc("inv_variance", 0);
  const bool center = ctx->Attr<bool>("center");
  const bool scale = ctx->Attr<bool>("scale");
  const int64_t begin_params_axis =
      ShiftNegativeAxisIfNeed(x.shape(), ctx->Attr<int64_t>("begin_params_axis"));
  y->set_shape(x.shape());
  y->set_is_dynamic(x.is_dynamic());
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
  mean->set_shape(InferBnParamShape(x.shape(), begin_norm_axis));
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
  user_op::TensorDesc* y = ctx->MutOutputTensorDesc("y", 0);
  y->set_data_type(x.data_type());
  if (center) {
    const user_op::TensorDesc& beta = ctx->InputTensorDesc("beta", 0);
    CHECK_EQ_OR_RETURN(beta.data_type(), x.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(x.data_type()) << ", but got "
        << DataType_Name(beta.data_type());
  }
  const bool scale = ctx->Attr<bool>("scale");
  if (scale) {
    const user_op::TensorDesc& gamma = ctx->InputTensorDesc("gamma", 0);
    CHECK_EQ_OR_RETURN(gamma.data_type(), x.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(x.data_type()) << ", but got "
        << DataType_Name(gamma.data_type());
  }
  user_op::TensorDesc* mean = ctx->MutOutputTensorDesc("mean", 0);
  user_op::TensorDesc* inv_variance = ctx->MutOutputTensorDesc("inv_variance", 0);
  mean->set_data_type(InferBnParamDataType(x.data_type()));
  inv_variance->set_data_type(mean->data_type());
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> LayerNormGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& mean = ctx->InputTensorDesc("mean", 0);
  const user_op::TensorDesc& inv_variance = ctx->InputTensorDesc("inv_variance", 0);
  user_op::TensorDesc* dx = ctx->MutOutputTensorDesc("dx", 0);
  CHECK_EQ_OR_RETURN(dy.shape(), x.shape());
  const int64_t begin_norm_axis = ctx->Attr<int64_t>("begin_norm_axis");
  CHECK_GT_OR_RETURN(begin_norm_axis, 0);
  const Shape& bn_param_shape = InferBnParamShape(x.shape(), begin_norm_axis);
  CHECK_EQ_OR_RETURN(mean.shape(), bn_param_shape);
  CHECK_EQ_OR_RETURN(inv_variance.shape(), bn_param_shape);
  dx->set_shape(dy.shape());
  dx->set_is_dynamic(dy.is_dynamic());
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
  CHECK_EQ_OR_RETURN(dy.data_type(), x.data_type())
      << "InferDataType Failed. Expected " << DataType_Name(x.data_type()) << ", but got "
      << DataType_Name(dy.data_type());
  const user_op::TensorDesc& mean = ctx->InputTensorDesc("mean", 0);
  const user_op::TensorDesc& inv_variance = ctx->InputTensorDesc("inv_variance", 0);
  DataType bn_param_data_type = InferBnParamDataType(x.data_type());
  CHECK_EQ_OR_RETURN(mean.data_type(), bn_param_data_type)
      << "InferDataType Failed. Expected " << DataType_Name(bn_param_data_type) << ", but got "
      << DataType_Name(mean.data_type());
  CHECK_EQ_OR_RETURN(inv_variance.data_type(), bn_param_data_type)
      << "InferDataType Failed. Expected " << DataType_Name(bn_param_data_type) << ", but got "
      << DataType_Name(inv_variance.data_type());
  user_op::TensorDesc* dx = ctx->MutOutputTensorDesc("dx", 0);
  dx->set_data_type(dy.data_type());
  if (ctx->has_input("_add_to_output", 0)) {
    const auto& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
    CHECK_EQ_OR_RETURN(add_to_output.data_type(), dx->data_type())
        << "InferDataType Failed. Expected " << DataType_Name(dx->data_type()) << ", but got "
        << DataType_Name(add_to_output.data_type());
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
    user_op::TensorDesc* beta_diff = ctx->MutOutputTensorDesc("beta_diff", 0);
    beta_diff->set_shape(param_shape);
  }
  if (has_gamma_diff) {
    user_op::TensorDesc* gamma_diff = ctx->MutOutputTensorDesc("gamma_diff", 0);
    gamma_diff->set_shape(param_shape);
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
    user_op::TensorDesc* beta_diff = ctx->MutOutputTensorDesc("beta_diff", 0);
    beta_diff->set_data_type(dy.data_type());
  }
  if (has_gamma_diff) {
    user_op::TensorDesc* gamma_diff = ctx->MutOutputTensorDesc("gamma_diff", 0);
    gamma_diff->set_data_type(dy.data_type());
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
