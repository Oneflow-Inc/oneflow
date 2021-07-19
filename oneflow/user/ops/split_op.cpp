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

        Maybe<void> InferTensorDesc(user_op::InferContext *ctx) {
            const auto dim = ctx->Attr<int64_t>("axis");
            const auto sections = ctx->Attr<int64_t>("sections");
            const user_op::TensorDesc &in_desc = ctx->InputTensorDesc("in", 0);
//  int64_t dynamic_dim_size = 0;
//  int64_t static_dim_size = 0;
            const int64_t in_num_axes = ctx->InputTensorDesc("in", 0).shape().NumAxes();
//  const int64_t like_num_axes = ctx->InputTensorDesc("like", 0).shape().NumAxes();
            CHECK_GE_OR_RETURN(dim, 0);
            CHECK_LT_OR_RETURN(dim, in_num_axes);
            printf("in inferTensorDesc\n");
            const int64_t min_split_size = in_desc.shape().At(dim) / sections;
            const int64_t num_splits_one_extra = in_desc.shape().At(dim) % sections;
            const int64_t num_splits = min_split_size + (num_splits_one_extra > 0 ? 1 : 0);
            printf("min_split:%d one:%d all:%d \n", min_split_size, num_splits_one_extra, num_splits);
            FOR_RANGE(int64_t, i, 0, num_splits)
            {
                printf("i:%d\n", i);
                //    const user_op::TensorDesc& like_i_desc = ctx->InputTensorDesc("like", i);
                user_op::TensorDesc *out_i_desc = ctx->OutputTensorDesc("out", i);
                //    CHECK_EQ_OR_RETURN(like_i_desc.shape().NumAxes(), like_num_axes);
                //    FOR_RANGE(int64_t, j, 0, like_num_axes) {
                //      if (j == axis) {
                //        if (like_i_desc.is_dynamic()) {
                //          dynamic_dim_size += like_i_desc.shape().At(j);
                //        } else {
                //          static_dim_size += like_i_desc.shape().At(j);
                //        }
                //      } else {
                //        CHECK_EQ_OR_RETURN(in_desc.shape().At(j), like_i_desc.shape().At(j));
                //      }
//            }
                DimVector out_i_dim_vec = in_desc.shape().dim_vec();
                out_i_dim_vec[dim] = i < min_split_size ? sections : num_splits_one_extra;
                for(auto it:out_i_dim_vec){
                    printf("it of out_dim:%d\n", it);
                }
//                FOR_RANGE(int64_t, j, 0, in_num_axes)
//                {
//                    if (dim != j)
//                        out_i_dim_vec.push_back(in_desc.shape().At(j));
//                    else
//                    {
//                        if(i < min_split_size)
//                            out_i_dim_vec.push_back(sections);
//                        else
//                            out_i_dim_vec.push_back(num_splits_one_extra);
//                    }
//
//                }
                printf("before mut\n");
                *out_i_desc->mut_shape() = Shape(out_i_dim_vec);
                printf("after mut\n");
            }
            printf("in inferTensorDesc\n");


//  if (dynamic_dim_size == 0) {
//    CHECK_EQ_OR_RETURN(static_dim_size, in_desc.shape().At(dim));
//  } else {
//    CHECK_LE_OR_RETURN(static_dim_size, in_desc.shape().At(dim));
//  }
            return Maybe<void>::Ok();
        }

        Maybe<void> InferDataType(user_op::InferContext *ctx) {
            const user_op::TensorDesc &in_desc = ctx->InputTensorDesc("in", 0);
            const auto dim = ctx->Attr<int64_t>("axis");
            const auto sections = ctx->Attr<int64_t>("sections");
            const int64_t min_split_size = in_desc.shape().At(dim) / sections;
            const int64_t num_splits_one_extra = in_desc.shape().At(dim) % sections;
            const int64_t num_splits = min_split_size + (num_splits_one_extra > 0 ? 1 : 0);
            FOR_RANGE(int64_t, i, 0, num_splits)
            {
                user_op::TensorDesc *out_i_desc = ctx->OutputTensorDesc("out", i);
                *out_i_desc->mut_data_type() = in_desc.data_type();
            }
            return Maybe<void>::Ok();
        }

//Maybe<void> SetLikeArgModifier(user_op::GetInputArgModifier GetInputArgModifierFn,
//                               const user_op::UserOpConfWrapper& user_op_conf) {
//  FOR_RANGE(int32_t, i, 0, user_op_conf.input_size("like")) {
//    user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", i);
//    CHECK_NOTNULL_OR_RETURN(like_modifier);
//    like_modifier->set_requires_grad(false);
//  }
//  return Maybe<void>::Ok();
//}

        Maybe<void> GetSbpSignature(user_op::SbpContext *ctx) {
            const auto dim = ctx->Attr<int64_t>("axis");
            const int64_t in_num_axes =
                    ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
//  const int64_t like_num_axes =
//      ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape().NumAxes();
            FOR_RANGE(int64_t, i, 0, in_num_axes)
            {
                if (i == dim) { continue; }
                ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
            }
            ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
//  std::vector<user_op::OpArg> like_arg_vec;
//  const size_t like_arg_size = ctx->outputs().size();
//  like_arg_vec.reserve(like_arg_size);
//  FOR_RANGE(int32_t, i, 0, like_arg_size) { like_arg_vec.emplace_back("like", i); }
//  FOR_RANGE(int64_t, i, like_num_axes, in_num_axes) {
//    ctx->NewBuilder()
//        .Split(user_op::OpArg("in", 0), i)
//        .Broadcast(like_arg_vec)
//        .Split(ctx->outputs(), i)
//        .Build();
//    ctx->NewBuilder()
//        .Split(user_op::OpArg("in", 0), i)
//        .PartialSum(like_arg_vec)
//        .Split(ctx->outputs(), i)
//        .Build();
//  }
//  ctx->NewBuilder()
//      .PartialSum(user_op::OpArg("in", 0))
//      .PartialSum(like_arg_vec)
//      .PartialSum(ctx->outputs())
//      .Build();
//  ctx->NewBuilder()
//      .PartialSum(user_op::OpArg("in", 0))
//      .Broadcast(like_arg_vec)
//      .PartialSum(ctx->outputs())
//      .Build();
//  ctx->NewBuilder()
//      .Broadcast(user_op::OpArg("in", 0))
//      .PartialSum(like_arg_vec)
//      .Broadcast(ctx->outputs())
//      .Build();
            return Maybe<void>::Ok();
        }

//        void GenGradOp(const user_op::UserOpWrapper &op, user_op::AddOpFn AddOp) {
//            const int64_t axis = op.attr<int64_t>("axis");
//            const int32_t out_size = op.output_size("out");
//            int64_t max_dim_size = 0;
//            FOR_RANGE(int32_t, i, 0, out_size)
//            {
//                max_dim_size += op.TensorDesc4ArgNameAndIndex("like", i).shape().At(axis);
//            }
//            if (op.NeedGenGradTensor4OpInput("in", 0)) {
//                user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
//                builder = builder.Op("concat");
//                FOR_RANGE(int32_t, i, 0, out_size)
//                {
//                    std::string out_diff_lbn;
//                    if (op.HasGradTensor4OpOutput("out", i)) {
//                        out_diff_lbn = op.GetGradTensorWithOpOutput("out", i);
//                    } else {
//                        auto zero_like_op = user_op::UserOpConfWrapperBuilder(op.op_name() + "_grad_zero_like_out_"
//                                                                              + std::to_string(i))
//                                .Op("zero_like")
//                                .Input("like", op.output("out", i))
//                                .Output("out")
//                                .Build();
//                        AddOp(zero_like_op);
//                        out_diff_lbn = zero_like_op.output("out", 0);
//                    }
//                    builder = builder.Input("in", out_diff_lbn);
//                }
//                user_op::UserOpConfWrapper grad_op =
//                        builder.Output("out").Attr("axis", axis).Attr("max_dim_size", max_dim_size).Build();
//
//                op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
//                AddOp(grad_op);
//            }
//        }

    }  // namespace

    REGISTER_USER_OP("split")
    .Input("in")
    .Output("out")
//    .InputWithMinimum("like", 2)
//    .OutputWithMinimum("out", 2)
    .Attr<int64_t>("axis")
    .Attr<int64_t>("sections")
    .SetTensorDescInferFn(InferTensorDesc)
//    .SetInputArgModifyFn(SetLikeArgModifier)
    .SetGetSbpFn(GetSbpSignature)
    .SetDataTypeInferFn(InferDataType);

//REGISTER_USER_OP_GRAD("split").SetGenBackwardOpConfFn(GenGradOp);

}  // namespace oneflow
