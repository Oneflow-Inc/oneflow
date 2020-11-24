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
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

REGISTER_USER_OP("partition")
    .Input("in")
    .Input("in_size")
    .OutputWithMinimum("out", 2)
    .OutputWithMinimum("out_size", 2)
    .Attr<int64_t>("parallel_num")
    .Attr<int64_t>("num_classes")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      const user_op::TensorDesc* in_size_desc = ctx->TensorDesc4ArgNameAndIndex("in_size", 0);
      const int64_t parallel_num = ctx->Attr<int64_t>("parallel_num");
      CHECK_EQ(ctx->user_op_conf().output_size("out"), parallel_num);
      CHECK_EQ(ctx->user_op_conf().output_size("out_size"), parallel_num);
      FOR_RANGE(int32_t, i, 0, parallel_num) {
        user_op::TensorDesc* out_i_desc = ctx->TensorDesc4ArgNameAndIndex("out", i);
        *out_i_desc = *in_desc;
        user_op::TensorDesc* out_size_desc = ctx->TensorDesc4ArgNameAndIndex("out_size", i);
        *out_size_desc = *in_size_desc;
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
