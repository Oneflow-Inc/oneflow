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

REGISTER_NO_GRAD_CPU_ONLY_USER_OP("megatron_gpt_mmap_data_loader")
    .OptionalInput("iteration")
    .Output("out")
    .Attr<std::string>("data_file_prefix")
    .Attr<int64_t>("seq_length")
    .Attr<int64_t>("label_length", 1)
    .Attr<int64_t>("num_samples")
    .Attr<int64_t>("batch_size")
    .Attr<DataType>("dtype")
    .Attr<std::vector<int64_t>>("split_sizes")
    .Attr<int64_t>("split_index")
    .Attr<bool>("shuffle")
    .Attr<int64_t>("random_seed")
    .Attr<std::vector<std::string>>("nd_sbp")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      int64_t batch_size = ctx->Attr<int64_t>("batch_size");
      int64_t sample_len = ctx->Attr<int64_t>("seq_length") + ctx->Attr<int64_t>("label_length");
      user_op::TensorDesc* out_desc = ctx->OutputTensorDesc("out", 0);
      *out_desc->mut_shape() = Shape({batch_size, sample_len});
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputTensorDesc("out", 0)->mut_data_type() = ctx->Attr<DataType>("dtype");
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          cfg::SbpParallel default_sbp;
          default_sbp.mutable_split_parallel()->set_axis(0);
          return user_op::InferNdSbp4SrcOp(ctx, default_sbp);
        })
    .SetInputArgModifyFn([](const user_op::GetInputArgModifier& GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper& conf) -> Maybe<void> {
      if (!conf.has_input("iteration", 0)) { return Maybe<void>::Ok(); }
      user_op::InputArgModifier* input_modifier = GetInputArgModifierFn("iteration", 0);
      CHECK_OR_RETURN(input_modifier != nullptr);
      input_modifier->set_is_mutable(true);
      input_modifier->set_requires_grad(false);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
