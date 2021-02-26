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

REGISTER_CPU_ONLY_USER_OP("OFRecordReader")
    .Output("out")
    .Attr<std::string>("data_dir")
    .Attr<int32_t>("data_part_num")
    .Attr<int32_t>("batch_size")
    .Attr<std::string>("part_name_prefix", "part-")
    .Attr<int32_t>("part_name_suffix_length", -1)
    .Attr<bool>("random_shuffle", false)
    .Attr<int64_t>("seed", -1)
    .Attr<int32_t>("shuffle_buffer_size", 1024)
    .Attr<bool>("shuffle_after_epoch", false)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      int32_t local_batch_size = ctx->Attr<int32_t>("batch_size");
      const SbpParallel& sbp = ctx->SbpParallel4ArgNameAndIndex("out", 0);
      int64_t parallel_num = ctx->parallel_ctx().parallel_num();
      if (sbp.has_split_parallel() && parallel_num > 1) {
        CHECK_EQ_OR_RETURN(local_batch_size % parallel_num, 0);
        local_batch_size /= parallel_num;
      }
      *out_tensor->mut_shape() = Shape({local_batch_size});
      *out_tensor->mut_data_type() = DataType::kOFRecord;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetOutputArgModifyFn([](user_op::GetOutputArgModifier GetOutputArgModifierFn,
                             const user_op::UserOpConfWrapper& conf) {
      user_op::OutputArgModifier* out_modifier = GetOutputArgModifierFn("out", 0);
      CHECK(out_modifier != nullptr);
      out_modifier->set_header_infered_before_compute(false);
    });

}  // namespace oneflow
