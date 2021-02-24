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

REGISTER_CPU_ONLY_USER_OP("bernoulli")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("seed", -1)
    .Attr<bool>("has_seed", false)
    .Attr<DataType>("dtype")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      *out_tensor->mut_shape() = in_tensor->shape();
      *out_tensor->mut_data_type() = ctx->Attr<DataType>("dtype");
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      for (int i = 0; i < in_tensor.shape().NumAxes(); ++i) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
