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

REGISTER_USER_OP("batch_gather")
    .Input("in")
    .Input("indices")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      CHECK_GT_OR_RETURN(in->shape().NumAxes(), 0);
      const user_op::TensorDesc* indices = ctx->TensorDesc4ArgNameAndIndex("indices", 0);
      CHECK_GT_OR_RETURN(indices->shape().NumAxes(), 0);
      CHECK_OR_RETURN(IsIndexDataType(indices->data_type()));
      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_LE_OR_RETURN(indices->shape().dim_vec().size(), in->shape().dim_vec().size());
      FOR_RANGE(int64_t, i, 0, indices->shape().dim_vec().size() - 1) {
        if (in->is_dynamic() && indices->is_dynamic() == false) {
          CHECK_GE_OR_RETURN(indices->shape().dim_vec().at(i), in->shape().dim_vec().at(i));
        } else if (in->is_dynamic() == false && indices->is_dynamic()) {
          UNIMPLEMENTED();
        } else {
          CHECK_EQ_OR_RETURN(indices->shape().dim_vec().at(i), in->shape().dim_vec().at(i));
        }
      }

      DimVector dim_vec(in->shape().dim_vec());
      dim_vec.at(indices->shape().NumAxes() - 1) = indices->shape().dim_vec().back();
      *out->mut_shape() = Shape(dim_vec);
      *out->mut_data_type() = in->data_type();
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("indices", 0);
      CHECK(indices_modifier != nullptr);
      indices_modifier->set_requires_grad(false);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const int64_t indices_num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0).shape().NumAxes();
      if (indices_num_axes > 1) {
        FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
          ctx->NewBuilder()
              .Split(user_op::OpArg("indices", 0), i)
              .Split(user_op::OpArg("in", 0), i)
              .Split(user_op::OpArg("out", 0), i)
              .Build();
        }
        ctx->NewBuilder()
            .Broadcast(user_op::OpArg("indices", 0))
            .PartialSum(user_op::OpArg("in", 0))
            .PartialSum(user_op::OpArg("out", 0))
            .Build();
      } else {
        std::shared_ptr<cfg::ErrorProto> err;
        err->set_msg("BatchGatherOp: indices_num_axes equals " + std::to_string(indices_num_axes)
                     + " (should be bigger than 1).");
        err->mutable_check_failed_error();
        return err;
      }

      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("batch_gather")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      bool need_grad_in = op.NeedGenGradTensor4OpInput("in", 0);
      if (need_grad_in) {
        const Shape in_shape = op.TensorDesc4ArgNameAndIndex("in", 0).shape();
        const Shape indices_shape = op.TensorDesc4ArgNameAndIndex("indices", 0).shape();

        user_op::UserOpConfWrapperBuilder in_grad_builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper in_grad_op =
            in_grad_builder.Op("unsorted_batch_segment_sum")
                .Input("data", op.GetGradTensorWithOpOutput("out", 0))
                .Input("segment_ids", op.input("indices", 0))
                .Output("out")
                .Attr("num_segments", in_shape.At(indices_shape.NumAxes() - 1))
                .Build();
        op.BindGradTensorWithOpInput(in_grad_op.output("out", 0), "in", 0);
        AddOp(in_grad_op);
      }
    });

}  // namespace oneflow
