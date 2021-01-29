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

REGISTER_USER_OP("expand")
    .Input("in")
    .Output("out")
    .Attr<std::vector<int32_t>>("in_shape")
    .Attr<std::vector<int32_t>>("out_shape")
    .Attr<std::vector<int32_t>>("stride")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      const auto& out_shape_attr = ctx->Attr<std::vector<int32_t>>("out_shape");
      DimVector dim_vec(out_shape_attr.begin(), out_shape_attr.end());
      *out_shape = Shape(dim_vec);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      const auto& in_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      const auto* in_batch_axis = ctx->BatchAxis4ArgNameAndIndex("in", 0);
      auto* out_batch_axis = ctx->BatchAxis4ArgNameAndIndex("out", 0);
      const auto& out_shape_attr = ctx->Attr<std::vector<int32_t>>("out_shape");
      const int32_t out_dims = out_shape_attr.size();
      const int32_t diff = out_dims - in_desc.shape().NumAxes();
      if (in_batch_axis->has_value()) {
        out_batch_axis->set_value(in_batch_axis->value() + diff);
      } else {
        out_batch_axis->clear_value();
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("in", 0))
          .PartialSum(user_op::OpArg("out", 0))
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("expand_grad")
    .Input("in")
    .Output("out")
    .Attr<std::vector<int32_t>>("out_shape")
    .Attr<std::vector<int32_t>>("stride")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      const auto& out_shape_attr = ctx->Attr<std::vector<int32_t>>("out_shape");
      DimVector dim_vec(out_shape_attr.begin(), out_shape_attr.end());
      *out_shape = Shape(dim_vec);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      const auto& in_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      const auto* in_batch_axis = ctx->BatchAxis4ArgNameAndIndex("in", 0);
      auto* out_batch_axis = ctx->BatchAxis4ArgNameAndIndex("out", 0);
      const auto& out_shape_attr = ctx->Attr<std::vector<int32_t>>("out_shape");
      const int32_t out_dims = out_shape_attr.size();
      const int32_t diff = in_desc.shape().NumAxes() - out_dims;
      if (in_batch_axis->has_value()) {
        out_batch_axis->set_value(in_batch_axis->value() - diff);
      } else {
        out_batch_axis->clear_value();
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("in", 0))
          .PartialSum(user_op::OpArg("out", 0))
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("expand").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op =
        builder.Op("expand_grad")
            .Input("in", op.GetGradTensorWithOpOutput("out", 0))
            .Output("out")
            .Attr<std::vector<int32_t>>("out_shape", op.attr<std::vector<int32_t>>("in_shape"))
            .Attr<std::vector<int32_t>>("stride", op.attr<std::vector<int32_t>>("stride"))
            .Build();
    op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
    AddOp(grad_op);
  }
});

}  // namespace oneflow
