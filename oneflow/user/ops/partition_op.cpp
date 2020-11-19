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
    .Input("in_num_unique")
    .OutputWithMinimum("out", 2)
    .OutputWithMinimum("num_unique", 2)
    .Attr<int64_t>("parallel_num")
    .Attr<int64_t>("num_classes")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      const user_op::TensorDesc* in_num_unique_desc =
          ctx->TensorDesc4ArgNameAndIndex("in_num_unique", 0);
      const int64_t parallel_num = ctx->Attr<int64_t>("parallel_num");
      // CHECK_EQ_OR_RETURN(parallel_num, ctx->outputs().size());
      FOR_RANGE(int32_t, i, 0, parallel_num) {
        user_op::TensorDesc* out_i_desc = ctx->TensorDesc4ArgNameAndIndex("out", i);
        *out_i_desc = *in_desc;
        user_op::TensorDesc* num_unique_desc = ctx->TensorDesc4ArgNameAndIndex("num_unique", i);
        *num_unique_desc = *in_num_unique_desc;
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
