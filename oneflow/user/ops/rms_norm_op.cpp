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

/* static */ Maybe<void> RmsNormOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& normalized_shape = ctx->Attr<Shape>("normalized_shape");
  if (ctx->has_input("weight", 0)) {
    const Shape& w_shape = ctx->InputShape("weight", 0);
    CHECK_EQ_OR_RETURN(w_shape, normalized_shape)
        << "expected weight shape " << normalized_shape.ToString() << ", got "
        << w_shape.ToString();
  }
  CHECK_LE_OR_RETURN(normalized_shape.size(), x_shape.size())
      << "invalid normalized shape " << normalized_shape.ToString() << " with input shape "
      << x_shape.ToString();
  size_t batch_ndim = x_shape.size() - normalized_shape.size();
  DimVector batch_dims(batch_ndim);
  for (int i = 0; i < x_shape.size(); ++i) {
    if (i < batch_ndim) {
      batch_dims[i] = x_shape[i];
    } else {
      CHECK_EQ_OR_RETURN(normalized_shape[i - batch_ndim], x_shape[i])
          << "invalid normalized shape " << normalized_shape.ToString() << " with input shape "
          << x_shape.ToString();
    }
  }
  ctx->SetOutputShape("y", 0, x_shape);
  ctx->SetOutputShape("inv_rms", 0, Shape{batch_dims});
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> RmsNormOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> RmsNormOp::InferDataType(user_op::InferContext* ctx) {
  DataType x_dtype = ctx->InputDType("x", 0);
  if (ctx->has_input("weight", 0)) {
    DataType w_dtype = ctx->InputDType("weight", 0);
    CHECK_EQ_OR_RETURN(w_dtype, x_dtype)
        << "RmsNormOp " << ctx->op_name() << " has different input dtype " << DataType_Name(x_dtype)
        << " and param dtype " << DataType_Name(w_dtype);
  }
  ctx->SetOutputDType("y", 0, x_dtype);

  DataType rms_dtype = x_dtype;
  if (x_dtype == DataType::kFloat16 || x_dtype == DataType::kBFloat16) {
    rms_dtype = DataType::kFloat;
  }
  ctx->SetOutputDType("inv_rms", 0, rms_dtype);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> RmsNormOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  const Shape& normalized_shape = ctx->Attr<Shape>("normalized_shape");
  size_t batch_ndim = x_shape.size() - normalized_shape.size();
  for (int i = 0; i < batch_ndim; ++i) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Broadcast(user_op::OpArg("weight", 0))
        .Split(ctx->outputs(), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> RmsNormGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& shape = ctx->InputShape("dy", 0);
  CHECK_EQ_OR_RETURN(ctx->InputShape("x", 0), shape);  // NOLINT(maybe-need-error-msg)
  // No need to check weight and inv_rms legality which should be guaranteed by forward op
  ctx->SetOutputShape("dx", 0, shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> RmsNormGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> RmsNormGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> RmsNormGradOp::GetSbp(user_op::SbpContext* ctx) {
  std::vector<user_op::OpArg> split_args = {user_op::OpArg("dy", 0), user_op::OpArg("x", 0),
                                            user_op::OpArg("inv_rms", 0)};
  std::vector<user_op::OpArg> broadcast_args;
  if (ctx->user_op_conf().has_input("weight", 0)) { broadcast_args.emplace_back("weight", 0); }
  const Shape& b_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("inv_rms", 0).shape();
  for (int i = 0; i < b_shape.size(); ++i) {
    ctx->NewBuilder()
        .Split(split_args, i)
        .Broadcast(broadcast_args)
        .Split(ctx->outputs(), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> RmsNormParamGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& shape = ctx->InputShape("dy", 0);
  CHECK_EQ_OR_RETURN(ctx->InputShape("x", 0), shape);  // NOLINT(maybe-need-error-msg)
  const Shape& b_shape = ctx->InputShape("inv_rms", 0);

  CHECK_LE_OR_RETURN(b_shape.size(), shape.size())
      << "invalid inv_rms shape " << b_shape.ToString() << " with dy shape " << shape.ToString();
  size_t n_ndim = shape.size() - b_shape.size();
  DimVector n_shape_vec(n_ndim);
  for (int i = 0; i < shape.size(); ++i) {
    if (i < b_shape.size()) {
      CHECK_EQ_OR_RETURN(b_shape[i], shape[i]) << "invalid inv_rms shape " << b_shape.ToString()
                                               << " with dy shape " << shape.ToString();
    } else {
      n_shape_vec[i - b_shape.size()] = shape[i];
    }
  }
  ctx->SetOutputShape("weight_grad", 0, Shape{n_shape_vec});
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> RmsNormParamGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> RmsNormParamGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("weight_grad", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> RmsNormParamGradOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& b_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("inv_rms", 0).shape();
  for (int i = 0; i < b_shape.size(); ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i).PartialSum(ctx->outputs()).Build();
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
