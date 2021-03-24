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

REGISTER_USER_OP("gather_dispatch")
    .Input("indices")
    .Output("idx")
    .OutputWithMinimum("out", 2)
    .OutputWithMinimum("count", 2)
    .Attr<int64_t>("parallel_num")
    .Attr<int64_t>("num_classes")
    .Attr<DataType>("dtype", DataType::kInt32)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* indices_desc = ctx->TensorDesc4ArgNameAndIndex("indices", 0);
      const int64_t num_classes = ctx->Attr<int64_t>("num_classes");
      const int64_t parallel_num = ctx->Attr<int64_t>("parallel_num");
      CHECK_EQ(num_classes % parallel_num, 0);
      const int64_t indices_elem_cnt = indices_desc->shape().elem_cnt();
      const DataType& data_type = ctx->Attr<DataType>("dtype");
      user_op::TensorDesc* idx_desc = ctx->TensorDesc4ArgNameAndIndex("idx", 0);
      *idx_desc = *indices_desc;
      *idx_desc->mut_data_type() = data_type;
      CHECK_EQ(ctx->user_op_conf().output_size("out"), parallel_num);
      CHECK_EQ(ctx->user_op_conf().output_size("count"), parallel_num);
      FOR_RANGE(int32_t, i, 0, parallel_num) {
        user_op::TensorDesc* out_i_desc = ctx->TensorDesc4ArgNameAndIndex("out", i);
        *out_i_desc = *indices_desc;
        *out_i_desc->mut_shape() = Shape({indices_elem_cnt});
        user_op::TensorDesc* count_desc = ctx->TensorDesc4ArgNameAndIndex("count", i);
        *count_desc->mut_shape() = Shape({1});
        *count_desc->mut_data_type() = data_type;
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
