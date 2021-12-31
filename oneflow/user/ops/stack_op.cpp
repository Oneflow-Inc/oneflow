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

// Maybe<void> GenGrapOp(const user_op::UserOpWrapper& op, const user_op::AddOpFn& AddOp) {
//   bool need_grad = false;
//   const int32_t in_size = op.input_size("in");
//   FOR_RANGE(int32_t, i, 0, in_size) {
//     if (op.NeedGenGradTensor4OpInput("in", i)) { need_grad = true; }
//   }
//   if (need_grad) {
//     user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
//     builder = builder.Op("split_like");
//     FOR_RANGE(int32_t, i, 0, in_size) { builder = builder.Input("like", op.input("in", i)); }
//     user_op::UserOpConfWrapper grad_op = builder.Input("in", op.GetGradTensorWithOpOutput("out", 0))
//                                              .Output("out", in_size)
//                                              .Attr("axis", op.attr<int64_t>("axis"))
//                                              .Build();

//     FOR_RANGE(int32_t, i, 0, in_size) {
//       if (op.NeedGenGradTensor4OpInput("in", i)) {
//         op.BindGradTensorWithOpInput(grad_op.output("out", i), "in", i);
//       }
//     }
//     AddOp(grad_op);
//   }
//   return Maybe<void>::Ok();
// }

}  // namespace

/* static */ Maybe<void> StackOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& first_in_desc = ctx->InputTensorDesc("in", 0);
  const int64_t axis = ctx->Attr<int64_t>("axis");
  CHECK_GE_OR_RETURN(axis, 0);
  const int64_t in_num_axes = first_in_desc.shape().NumAxes(); 
  CHECK_LE_OR_RETURN(axis, in_num_axes);
  DimVector out_dim_vec(in_num_axes+1); 
  for(int i = 0; i < in_num_axes+1; i++){
      if(i == axis){
        continue; 
      }else if(i < axis){
        out_dim_vec.at(i) = first_in_desc.shape().At(i);
      }else{
        out_dim_vec.at(i) = first_in_desc.shape().At(i-1);
      }
  }
  printf("log here \n"); 
//   out_dim_vec.at(axis) = 0;
  int64_t dynamic_dim_size = 0;
  for (const auto& in_arg_pair : ctx->inputs()) {
    const user_op::TensorDesc& in_desc =
        ctx->InputTensorDesc(in_arg_pair.first, in_arg_pair.second);
    CHECK_EQ_OR_RETURN(in_desc.shape().NumAxes(), first_in_desc.shape().NumAxes());
    FOR_RANGE(int64_t, i, 0, in_num_axes+1) {
      if (i == axis) {
        if (in_desc.is_dynamic()) {
          dynamic_dim_size += in_desc.shape().At(i); // I dont know!
        } else {
          out_dim_vec.at(axis) += 1; 
        }
      } else if(i < axis) {
        CHECK_EQ_OR_RETURN(in_desc.shape().At(i), out_dim_vec.at(i));
      }else{
        CHECK_EQ_OR_RETURN(in_desc.shape().At(i-1), out_dim_vec.at(i));
      }
    }
  }
  printf("HERE \n"); 
  user_op::TensorDesc* out_desc = ctx->OutputTensorDesc("out", 0);
  const int64_t max_dim_size = ctx->Attr<int64_t>("max_dim_size");
//   CHECK_LE_OR_RETURN(out_dim_vec.at(axis), max_dim_size); // i dont know
  if (dynamic_dim_size == 0) {
    out_desc->set_is_dynamic(false);
  } else {
    out_desc->set_is_dynamic(true);
    out_dim_vec.at(axis) = max_dim_size;
  }
  *out_desc->mut_shape() = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> StackOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}


// todo!
/* static */ Maybe<void> StackOp::GetSbp(user_op::SbpContext* ctx) {
  const int64_t axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc& first_in_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, first_in_desc.shape().NumAxes()) {
    if (i == axis) { continue; }
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> StackOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& first_in_desc = ctx->InputTensorDesc("in", 0);
  for (const auto& in_arg_pair : ctx->inputs()) {
    const user_op::TensorDesc& in_desc =
        ctx->InputTensorDesc(in_arg_pair.first, in_arg_pair.second);
    CHECK_EQ_OR_RETURN(in_desc.data_type(), first_in_desc.data_type());
  }
  user_op::TensorDesc* out_desc = ctx->OutputTensorDesc("out", 0);
  *out_desc->mut_data_type() = first_in_desc.data_type();
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> StackOp::CheckAttr(const user_op::UserOpDefWrapper&,
                                           const user_op::UserOpConfWrapper& op_conf) {
  CHECK_OR_RETURN(op_conf.input_size("in") >= 2);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> StackBackwardOp::GetSbp(user_op::SbpContext* ctx) {
  // todo FIX!

  const auto axis = ctx->Attr<int64_t>("axis");
  const int64_t in_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
  const int64_t like_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, like_num_axes) {
    if (i == axis) { continue; }
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  std::vector<user_op::OpArg> like_arg_vec;
  const size_t like_arg_size = ctx->outputs().size();
  like_arg_vec.reserve(like_arg_size);
  FOR_RANGE(int32_t, i, 0, like_arg_size) { like_arg_vec.emplace_back("like", i); }
  FOR_RANGE(int64_t, i, like_num_axes, in_num_axes) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), i)
        .Broadcast(like_arg_vec)
        .Split(ctx->outputs(), i)
        .Build();
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), i)
        .PartialSum(like_arg_vec)
        .Split(ctx->outputs(), i)
        .Build();
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
/*static*/ Maybe<void> StackBackwardOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  int64_t dynamic_dim_size = 0;
  int64_t static_dim_size = 0;
  const int64_t in_num_axes = ctx->InputTensorDesc("in", 0).shape().NumAxes();
  const int64_t like_num_axes = ctx->InputTensorDesc("like", 0).shape().NumAxes();
  CHECK_LE_OR_RETURN(like_num_axes, in_num_axes);
  CHECK_LT_OR_RETURN(axis, like_num_axes);
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    const user_op::TensorDesc& like_i_desc = ctx->InputTensorDesc("like", i);
    user_op::TensorDesc* out_i_desc = ctx->OutputTensorDesc("out", i);
    CHECK_EQ_OR_RETURN(like_i_desc.shape().NumAxes(), like_num_axes);
    FOR_RANGE(int64_t, j, 0, like_num_axes+1) {
      if (j == axis) {
        if (like_i_desc.is_dynamic()) {
          dynamic_dim_size += like_i_desc.shape().Count(j);
        } else {
          static_dim_size += like_i_desc.shape().Count(j);
        }
      } else if (j < axis){
        CHECK_EQ_OR_RETURN(in_desc.shape().At(j), like_i_desc.shape().At(j));
      } else {
        CHECK_EQ_OR_RETURN(in_desc.shape().At(j), like_i_desc.shape().At(j-1));
      }
    }
    DimVector out_i_dim_vec = like_i_desc.shape().dim_vec();
    // No need ?
    // FOR_RANGE(int64_t, j, like_num_axes, in_num_axes) {
    //   out_i_dim_vec.emplace_back(in_desc.shape().At(j));
    // }
    *out_i_desc->mut_shape() = Shape(out_i_dim_vec);
    out_i_desc->set_is_dynamic(like_i_desc.is_dynamic());
  }
  if (dynamic_dim_size == 0) {
    CHECK_EQ_OR_RETURN(static_dim_size, in_desc.shape().Count(axis));
  } else {
    CHECK_LE_OR_RETURN(static_dim_size, in_desc.shape().Count(axis));
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> StackBackwardOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> StackBackwardOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    user_op::TensorDesc* out_i_desc = ctx->OutputTensorDesc("out", i);
    *out_i_desc->mut_data_type() = in_desc.data_type();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> StackBackwardOp::ModifyInputArg(const GetInputArgModifier& GetInputArgModifierFn,
                                                   const user_op::UserOpConfWrapper& user_op_conf) {
  FOR_RANGE(int32_t, i, 0, user_op_conf.input_size("like")) {
    user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", i);
    CHECK_NOTNULL_OR_RETURN(like_modifier);
    like_modifier->set_requires_grad(false);
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> StackBackwardOp::CheckAttr(const user_op::UserOpDefWrapper&,
                                              const user_op::UserOpConfWrapper& op_conf) {
  CHECK_OR_RETURN(op_conf.input_size("like") >= 2);
  CHECK_OR_RETURN(op_conf.output_size("out") >= 2);
  return Maybe<void>::Ok();
}

// REGISTER_USER_OP_GRAD("Stack").SetGenBackwardOpConfFn(GenGrapOp);

}  // namespace oneflow
