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

REGISTER_USER_OP("unique_with_counts")
    .Input("x")
    .Output("y")
    .Output("idx")
    .Output("count")
    .Output("num_unique")
    .Attr<DataType>("out_idx", DataType::kInt32)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      CHECK_EQ_OR_RETURN(x->shape().NumAxes(), 1);

      user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      *y->mut_shape() = x->shape();
      *y->mut_is_dynamic() = x->is_dynamic();

      user_op::TensorDesc* idx = ctx->TensorDesc4ArgNameAndIndex("idx", 0);
      *idx->mut_shape() = x->shape();
      *idx->mut_is_dynamic() = x->is_dynamic();

      user_op::TensorDesc* count = ctx->TensorDesc4ArgNameAndIndex("count", 0);
      *count->mut_shape() = x->shape();
      *count->mut_is_dynamic() = x->is_dynamic();

      user_op::TensorDesc* num_unique = ctx->TensorDesc4ArgNameAndIndex("num_unique", 0);
      *num_unique->mut_shape() = Shape({1});
      return Maybe<void>::Ok();
    })
    .SetInferDataTypeFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      auto out_idx = ctx->Attr<DataType>("out_idx");
      CHECK_OR_RETURN(IsIndexDataType(out_idx));
      user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      *y->mut_data_type() = x->data_type();

      user_op::TensorDesc* idx = ctx->TensorDesc4ArgNameAndIndex("idx", 0);
      *idx->mut_data_type() = x->data_type();
      *idx->mut_data_type() = out_idx;

      user_op::TensorDesc* count = ctx->TensorDesc4ArgNameAndIndex("count", 0);
      *count->mut_data_type() = out_idx;
      user_op::TensorDesc* num_unique = ctx->TensorDesc4ArgNameAndIndex("num_unique", 0);
      *num_unique->mut_data_type() = out_idx;
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
