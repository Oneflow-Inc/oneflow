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

/* static */ Maybe<void> StackOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& first_in_desc = ctx->InputTensorDesc("in", 0);
  const int64_t axis = ctx->Attr<int64_t>("axis");
  CHECK_GE_OR_RETURN(axis, 0) << "The axis should be greater than or equal to 0.";
  const int64_t in_num_axes = first_in_desc.shape().NumAxes();
  CHECK_LE_OR_RETURN(axis, in_num_axes)
      << "The axis should be less than or equal to input num axes.";
  DimVector out_dim_vec(in_num_axes + 1);
  for (int i = 0; i < in_num_axes + 1; i++) {
    if (i == axis) {
      continue;
    } else if (i < axis) {
      out_dim_vec.at(i) = first_in_desc.shape().At(i);
    } else {
      out_dim_vec.at(i) = first_in_desc.shape().At(i - 1);
    }
  }
  int64_t dynamic_dim_size = 0;
  for (const auto& in_arg_pair : ctx->inputs()) {
    const user_op::TensorDesc& in_desc =
        ctx->InputTensorDesc(in_arg_pair.first, in_arg_pair.second);
    CHECK_EQ_OR_RETURN(in_desc.shape().NumAxes(), first_in_desc.shape().NumAxes())
        << "The num axes of input should be equal to first input's num axes. ";
    FOR_RANGE(int64_t, i, 0, in_num_axes + 1) {
      if (i == axis) {
        if (in_desc.is_dynamic()) {
          dynamic_dim_size += 1;
        } else {
          out_dim_vec.at(axis) += 1;
        }
      } else if (i < axis) {
        CHECK_EQ_OR_RETURN(in_desc.shape().At(i), out_dim_vec.at(i))
            << "The input shape at axis " << i << " is not equal to out shape at axis " << i;
      } else {
        CHECK_EQ_OR_RETURN(in_desc.shape().At(i - 1), out_dim_vec.at(i))
            << "The input shape at axis " << i - 1 << " is not equal to out shape at axis " << i;
      }
    }
  }
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  const int64_t max_dim_size = ctx->Attr<int64_t>("max_dim_size");
  CHECK_LE_OR_RETURN(out_dim_vec.at(axis), max_dim_size)
      << "The out shape at axis " << axis << " should be less equal to " << max_dim_size;
  if (dynamic_dim_size == 0) {
    out_desc->set_is_dynamic(false);
  } else {
    out_desc->set_is_dynamic(true);
    out_dim_vec.at(axis) = max_dim_size;
  }
  out_desc->set_shape(Shape(out_dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> StackOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> StackOp::GetSbp(user_op::SbpContext* ctx) {
  const int64_t axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc& first_in_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, first_in_desc.shape().NumAxes()) {
    /*
    Stack can be view as expand_dims + concat.
    For stack([(2, 4, 6), (2, 4, 6), axis=1]), it equals to [2, 4, 6]->[2, 1, 4, 6]. concat([2, 1,
    4, 6], [2, 1, 4, 6], concat_dim=1) Concat split all the axis except the concat_dim.
    */
    if (i >= axis) {
      ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i + 1).Build();
    } else {
      ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
    }
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> StackOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& first_in_desc = ctx->InputTensorDesc("in", 0);
  for (const auto& in_arg_pair : ctx->inputs()) {
    const user_op::TensorDesc& in_desc =
        ctx->InputTensorDesc(in_arg_pair.first, in_arg_pair.second);
    CHECK_EQ_OR_RETURN(in_desc.data_type(), first_in_desc.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(first_in_desc.data_type())
        << ", but got " << DataType_Name(in_desc.data_type());
  }
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_data_type(first_in_desc.data_type());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> StackOp::CheckAttr(const user_op::UserOpDefWrapper&,
                                          const user_op::UserOpConfWrapper& op_conf) {
  CHECK_OR_RETURN(op_conf.input_size("in") >= 1)
      << "The size of input should be greater than or equal to 1. ";
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> StackGradOp::GetSbp(user_op::SbpContext* ctx) {
  const auto axis = ctx->Attr<int64_t>("axis");
  const int64_t like_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape().NumAxes();
  std::vector<user_op::OpArg> like_arg_vec;
  const size_t like_arg_size = ctx->outputs().size();
  like_arg_vec.reserve(like_arg_size);
  FOR_RANGE(int32_t, i, 0, like_arg_size) { like_arg_vec.emplace_back("like", i); }
  FOR_RANGE(int64_t, i, 0, like_num_axes) {
    if (i >= axis) {
      ctx->NewBuilder()
          .Split(like_arg_vec, i)
          .Split(ctx->outputs(), i)
          .Split(user_op::OpArg("in", 0), i + 1)
          .Build();
    } else {
      ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
    }
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(like_arg_vec)
      .PartialSum(ctx->outputs())
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .Broadcast(like_arg_vec)
      .PartialSum(ctx->outputs())
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("in", 0))
      .PartialSum(like_arg_vec)
      .Broadcast(ctx->outputs())
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> StackGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  int64_t dynamic_dim_size = 0;
  int64_t static_dim_size = 0;
  const int64_t in_num_axes = ctx->InputTensorDesc("in", 0).shape().NumAxes();
  const int64_t like_num_axes = ctx->InputTensorDesc("like", 0).shape().NumAxes();
  CHECK_LE_OR_RETURN(like_num_axes, in_num_axes)
      << "The num axes of `like` tensor should be less equal to num axes of `in` tensor. ";
  CHECK_LE_OR_RETURN(axis, like_num_axes)
      << "The axis should be less equal than num axes of `like` tensor. ";
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    const user_op::TensorDesc& like_i_desc = ctx->InputTensorDesc("like", i);
    user_op::TensorDesc* out_i_desc = ctx->MutOutputTensorDesc("out", i);
    CHECK_EQ_OR_RETURN(like_i_desc.shape().NumAxes(), like_num_axes)
        << "The num axes of `like` tensor at index " << i
        << " should be equal to first `like` tensor. ";
    FOR_RANGE(int64_t, j, 0, like_num_axes + 1) {
      if (j == axis) {
        if (like_i_desc.is_dynamic()) {
          dynamic_dim_size += like_i_desc.shape().Count(j);
        } else {
          static_dim_size += like_i_desc.shape().Count(j);
        }
      } else if (j < axis) {
        CHECK_EQ_OR_RETURN(in_desc.shape().At(j), like_i_desc.shape().At(j))
            << " Stack Grad expects the shape of input tensor is equal to like tensor's. "
               ", but got "
            << in_desc.shape().ToString() << " at input and " << like_i_desc.shape().ToString()
            << "at like ";
      } else {
        CHECK_EQ_OR_RETURN(in_desc.shape().At(j), like_i_desc.shape().At(j - 1))
            << " Stack Grad expects the shape of input tensor is equal to like tensor's. "
               ", but got "
            << in_desc.shape().ToString() << " at input and " << like_i_desc.shape().ToString()
            << "at like ";
      }
    }
    DimVector out_i_dim_vec = like_i_desc.shape().dim_vec();
    out_i_desc->set_shape(Shape(out_i_dim_vec));
    out_i_desc->set_is_dynamic(like_i_desc.is_dynamic());
  }
  if (dynamic_dim_size == 0) {
    CHECK_EQ_OR_RETURN(static_dim_size, in_desc.shape().Count(axis))
        << "In non dynamic shape situation, the static dim size should be equal to input tensor "
           "size. ";
  } else {
    CHECK_LE_OR_RETURN(static_dim_size, in_desc.shape().Count(axis))
        << "In dynamic shape situation, the static dim size should be less equal to input tensor "
           "size. ";
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> StackGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> StackGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    user_op::TensorDesc* out_i_desc = ctx->MutOutputTensorDesc("out", i);
    out_i_desc->set_data_type(in_desc.data_type());
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> StackGradOp::ModifyInputArg(const GetInputArgModifier& GetInputArgModifierFn,
                                                   const user_op::UserOpConfWrapper& user_op_conf) {
  FOR_RANGE(int32_t, i, 0, user_op_conf.input_size("like")) {
    user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", i);
    CHECK_NOTNULL_OR_RETURN(like_modifier);
    like_modifier->set_requires_grad(false);
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> StackGradOp::CheckAttr(const user_op::UserOpDefWrapper&,
                                              const user_op::UserOpConfWrapper& op_conf) {
  CHECK_OR_RETURN(op_conf.input_size("like") >= 1)
      << "The count of like tensor should be greater than or equal to 1. ";
  CHECK_OR_RETURN(op_conf.output_size("out") >= 1)
      << "The count of out tensor should be greater than or equal to 1. ";
  return Maybe<void>::Ok();
}

}  // namespace oneflow
