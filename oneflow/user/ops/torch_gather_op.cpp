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

REGISTER_USER_OP("torch_gather")
    .Input("input")
    .Input("index")
    .Output("out")
    .Attr("dim", UserOpAttrType::kAtInt64)
    .Attr("sparse_grad", UserOpAttrType::kAtBool)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("input", 0);
      CHECK_GT_OR_RETURN(in->shape().NumAxes(), 0);
      const int64_t axis = ctx->Attr<int64_t>("dim");
      const user_op::TensorDesc* indices = ctx->TensorDesc4ArgNameAndIndex("index", 0);
      CHECK_OR_RETURN(IsIndexDataType(indices->data_type()));
      CHECK_GT_OR_RETURN(indices->shape().NumAxes(), 0);
      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);

      DimVector dim_vec;
      dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin(),
                     in->shape().dim_vec().cbegin() + axis);
      dim_vec.insert(dim_vec.end(), indices->shape().dim_vec().cbegin(),
                     indices->shape().dim_vec().cend());
      dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin() + axis + 1,
                     in->shape().dim_vec().end());
      *out->mut_shape() = Shape(dim_vec);
      out->set_is_dynamic(indices->is_dynamic());
      *out->mut_data_type() = in->data_type();
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("indices", 0);
      CHECK(indices_modifier != nullptr);
      indices_modifier->set_requires_grad(false);
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      if (ctx->BatchAxis4ArgNameAndIndex("index", 0)->has_value()) {
        ctx->BatchAxis4ArgNameAndIndex("out", 0)->set_value(
            ctx->Attr<int64_t>("dim") + ctx->BatchAxis4ArgNameAndIndex("index", 0)->value());
      } else {
        ctx->BatchAxis4ArgNameAndIndex("out", 0)->clear_value();
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const int64_t in_num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0).shape().NumAxes();
      const int64_t indices_num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("index", 0).shape().NumAxes();
      const int64_t gather_axis = ctx->Attr<int64_t>("dim");
      CHECK_GE_OR_RETURN(gather_axis, 0);
      CHECK_LT_OR_RETURN(gather_axis, in_num_axes);
      FOR_RANGE(int64_t, i, 0, indices_num_axes) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("index", 0), i)
            .Broadcast(user_op::OpArg("input", 0))
            .Split(user_op::OpArg("out", 0), gather_axis + i)
            .Build();
      }
      FOR_RANGE(int64_t, i, 0, in_num_axes) {
        if (i == gather_axis) {
          ctx->NewBuilder()
              .Broadcast(user_op::OpArg("index", 0))
              .Split(user_op::OpArg("input", 0), i)
              .PartialSum(user_op::OpArg("out", 0))
              .Build();
        } else {
          ctx->NewBuilder()
              .Broadcast(user_op::OpArg("index", 0))
              .Split(user_op::OpArg("input", 0), i)
              .Split(user_op::OpArg("out", 0), i < gather_axis ? i : i + indices_num_axes - 1)
              .Build();
        }
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("torch_gather").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                              user_op::AddOpFn AddOp) {
    // to be added...
  });

}  // namespace oneflow
