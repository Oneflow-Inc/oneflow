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

/* static */ Maybe<void> ConcatOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& first_in_desc = ctx->InputTensorDesc("in", 0);
  const int64_t axis = ctx->Attr<int64_t>("axis");
  CHECK_GE_OR_RETURN(axis, 0);
  CHECK_LT_OR_RETURN(axis, first_in_desc.shape().NumAxes());
  DimVector out_dim_vec = first_in_desc.shape().dim_vec();
  out_dim_vec.at(axis) = 0;
  int64_t first_axes = first_in_desc.shape().NumAxes();
  int64_t first_elemcnt = first_in_desc.shape().elem_cnt();
  int64_t dynamic_dim_size = 0;
  for (const auto& in_arg_pair : ctx->inputs()) {
    const user_op::TensorDesc& in_desc =
        ctx->InputTensorDesc(in_arg_pair.first, in_arg_pair.second);
    if (first_elemcnt == 0 and first_axes == 1) {
      if (in_desc.shape().elem_cnt() != 0 or in_desc.shape().NumAxes() != 1) {
        out_dim_vec = in_desc.shape().dim_vec();
        out_dim_vec.at(axis) = 0;
        first_axes = in_desc.shape().NumAxes();
        first_elemcnt = in_desc.shape().elem_cnt();
      } else {
        continue;
      }
    } else if (in_desc.shape().elem_cnt() != 0 or in_desc.shape().NumAxes() != 1) {
      CHECK_EQ_OR_RETURN(in_desc.shape().NumAxes(), first_axes);
    }
    FOR_RANGE(int64_t, i, 0, in_desc.shape().NumAxes()) {
      if (in_desc.shape().elem_cnt() == 0 and in_desc.shape().NumAxes() == 1) { continue; }
      if (i == axis) {
        if (in_desc.is_dynamic()) {
          dynamic_dim_size += in_desc.shape().At(i);
        } else {
          out_dim_vec.at(axis) += in_desc.shape().At(i);
        }
      } else {
        CHECK_EQ_OR_RETURN(in_desc.shape().At(i), out_dim_vec.at(i));
      }
    }
  }

  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  const int64_t max_dim_size = ctx->Attr<int64_t>("max_dim_size");
  CHECK_LE_OR_RETURN(out_dim_vec.at(axis), max_dim_size);
  if (dynamic_dim_size == 0) {
    out_desc->set_is_dynamic(false);
  } else {
    out_desc->set_is_dynamic(true);
    out_dim_vec.at(axis) = max_dim_size;
  }
  out_desc->set_shape(Shape(out_dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ConcatOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ConcatOp::GetSbp(user_op::SbpContext* ctx) {
  const int64_t axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc& first_in_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, first_in_desc.shape().NumAxes()) {
    if (i == axis) { continue; }
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ConcatOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& first_in_desc = ctx->InputTensorDesc("in", 0);
  for (const auto& in_arg_pair : ctx->inputs()) {
    const user_op::TensorDesc& in_desc =
        ctx->InputTensorDesc(in_arg_pair.first, in_arg_pair.second);
    CHECK_EQ_OR_RETURN(in_desc.data_type(), first_in_desc.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(in_desc.data_type()) << ", but got "
        << DataType_Name(first_in_desc.data_type());
  }
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_data_type(first_in_desc.data_type());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ConcatOp::CheckAttr(const user_op::UserOpDefWrapper&,
                                           const user_op::UserOpConfWrapper& op_conf) {
  CHECK_OR_RETURN(op_conf.input_size("in") >= 2);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
