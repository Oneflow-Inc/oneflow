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

REGISTER_USER_OP("triu")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("diagonal")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
      user_op::TensorDesc* out = ctx->OutputTensorDesc("out", 0);
      CHECK_GE_OR_RETURN(in.shape().NumAxes(), 2);
      *out->mut_shape() = in.shape();
      *out->mut_is_dynamic() = in.is_dynamic();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
      user_op::TensorDesc* out = ctx->OutputTensorDesc("out", 0);
      *out->mut_data_type() = in.data_type();
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, i, 0, in.shape().NumAxes() - 2) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("in", 0))
          .PartialSum(user_op::OpArg("out", 0))
          .Build();
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
