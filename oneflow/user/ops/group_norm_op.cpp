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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

oneflow::DataType InferGnParamDataType(const DataType x_data_type) {
  return (x_data_type == DataType::kFloat16 || x_data_type == DataType::kBFloat16)
             ? DataType::kFloat
             : x_data_type;
}

}  // namespace

/* static */ Maybe<void> GroupNormOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  user_op::TensorDesc* y = ctx->MutOutputTensorDesc("y", 0);
  user_op::TensorDesc* mean = ctx->MutOutputTensorDesc("mean", 0);
  user_op::TensorDesc* inv_variance = ctx->MutOutputTensorDesc("inv_variance", 0);
  const bool affine = ctx->Attr<bool>("affine");
  const int32_t num_groups = ctx->Attr<int32_t>("num_groups");
  const int64_t batch_size = x.shape().At(0);
  const std::string& data_format = ctx->Attr<std::string>("data_format");
  CHECK_GT_OR_RETURN(x.shape().NumAxes(), 2);
  int64_t channel_size = 0;
  if (data_format == "channels_first") {
    channel_size = x.shape().At(1);
  } else if (data_format == "channels_last") {
    channel_size = x.shape().At(x.shape().NumAxes() - 1);
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  y->set_shape(x.shape());
  y->set_is_dynamic(x.is_dynamic());
  if (affine) {
    const user_op::TensorDesc& gamma = ctx->InputTensorDesc("gamma", 0);
    CHECK_EQ_OR_RETURN(gamma.shape().At(0), channel_size);
    const user_op::TensorDesc& beta = ctx->InputTensorDesc("beta", 0);
    CHECK_EQ_OR_RETURN(beta.shape().At(0), channel_size);
  }
  CHECK_EQ_OR_RETURN(channel_size % num_groups, 0) << "Channels should be divisble by num_groups. ";
  mean->set_shape(Shape({batch_size, num_groups}));
  *inv_variance = *mean;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> GroupNormOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> GroupNormOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(ctx->inputs(), 0)
      .Split(ctx->outputs(), 0)
      .Broadcast(user_op::OpArg("gamma", 0))
      .Broadcast(user_op::OpArg("beta", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> GroupNormOp::InferDataType(user_op::InferContext* ctx) {
  const bool affine = ctx->Attr<bool>("affine");
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  user_op::TensorDesc* y = ctx->MutOutputTensorDesc("y", 0);
  y->set_data_type(x.data_type());
  if (affine) {
    const user_op::TensorDesc& gamma = ctx->InputTensorDesc("gamma", 0);
    CHECK_EQ_OR_RETURN(gamma.data_type(), x.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(x.data_type()) << ", but got "
        << DataType_Name(gamma.data_type());
    const user_op::TensorDesc& beta = ctx->InputTensorDesc("beta", 0);
    CHECK_EQ_OR_RETURN(beta.data_type(), x.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(x.data_type()) << ", but got "
        << DataType_Name(beta.data_type());
  }
  user_op::TensorDesc* mean = ctx->MutOutputTensorDesc("mean", 0);
  user_op::TensorDesc* inv_variance = ctx->MutOutputTensorDesc("inv_variance", 0);
  mean->set_data_type(InferGnParamDataType(x.data_type()));
  inv_variance->set_data_type(mean->data_type());
  return Maybe<void>::Ok();
}

// GroupNorm Grad
/* static */ Maybe<void> GroupNormGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& mean = ctx->InputTensorDesc("mean", 0);
  const user_op::TensorDesc& inv_variance = ctx->InputTensorDesc("inv_variance", 0);
  const int32_t num_groups = ctx->Attr<int32_t>("num_groups");
  user_op::TensorDesc* dx = ctx->MutOutputTensorDesc("dx", 0);
  CHECK_EQ_OR_RETURN(dy.shape(), x.shape());
  const Shape& gn_param_shape = Shape({x.shape().At(0), num_groups});
  CHECK_EQ_OR_RETURN(mean.shape(), gn_param_shape);
  CHECK_EQ_OR_RETURN(inv_variance.shape(), gn_param_shape);
  dx->set_shape(dy.shape());
  dx->set_is_dynamic(dy.is_dynamic());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> GroupNormGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> GroupNormGradOp::GetSbp(user_op::SbpContext* ctx) {
  std::vector<user_op::OpArg> broadcast_args;
  if (ctx->user_op_conf().has_input("gamma", 0)) {
    broadcast_args.emplace_back(user_op::OpArg("gamma", 0));
  }

  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("mean", 0), 0)
      .Split(user_op::OpArg("inv_variance", 0), 0)
      .Split(ctx->outputs(), 0)
      .Broadcast(broadcast_args)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> GroupNormGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  CHECK_EQ_OR_RETURN(dy.data_type(), x.data_type())
      << "InferDataType Failed. Expected " << DataType_Name(x.data_type()) << ", but got "
      << DataType_Name(dy.data_type());
  const user_op::TensorDesc& mean = ctx->InputTensorDesc("mean", 0);
  const user_op::TensorDesc& inv_variance = ctx->InputTensorDesc("inv_variance", 0);
  const DataType& gn_param_data_type = InferGnParamDataType(x.data_type());
  CHECK_EQ_OR_RETURN(mean.data_type(), gn_param_data_type)
      << "InferDataType Failed. Expected " << DataType_Name(gn_param_data_type) << ", but got "
      << DataType_Name(mean.data_type());
  CHECK_EQ_OR_RETURN(inv_variance.data_type(), gn_param_data_type)
      << "InferDataType Failed. Expected " << DataType_Name(gn_param_data_type) << ", but got "
      << DataType_Name(inv_variance.data_type());
  user_op::TensorDesc* dx = ctx->MutOutputTensorDesc("dx", 0);
  dx->set_data_type(dy.data_type());
  return Maybe<void>::Ok();
}

// GroupNorm Param Grad
/* static */ Maybe<void> GroupNormParamGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  user_op::TensorDesc* dgamma = ctx->MutOutputTensorDesc("dgamma", 0);
  user_op::TensorDesc* dbeta = ctx->MutOutputTensorDesc("dbeta", 0);
  const int64_t channel_size = x.shape().At(1);
  dgamma->set_shape(Shape{channel_size});
  dbeta->set_shape(Shape{channel_size});
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> GroupNormParamGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> GroupNormParamGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("mean", 0), 0)
      .Split(user_op::OpArg("inv_variance", 0), 0)
      .PartialSum(ctx->outputs())
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> GroupNormParamGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  user_op::TensorDesc* dgamma = ctx->MutOutputTensorDesc("dgamma", 0);
  user_op::TensorDesc* dbeta = ctx->MutOutputTensorDesc("dbeta", 0);
  dgamma->set_data_type(dy.data_type());
  dbeta->set_data_type(dy.data_type());
  return Maybe<void>::Ok();
}

}  // namespace oneflow
