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
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {

REGISTER_CPU_ONLY_USER_OP("gpt_data_loader")
    .Input("iteration")
    .Output("sequence")
    .Attr<std::string>("data_file_prefix")
    .Attr<int64_t>("seq_length")
    .Attr<int64_t>("num_samples")
    .Attr<int64_t>("batch_size")
    .Attr<DataType>("dtype")
    .Attr<std::vector<int64_t>>("split_sizes")
    .Attr<int64_t>("split_index")
    .Attr<bool>("shuffle")
    .Attr<int64_t>("random_seed")
    .Attr<std::vector<std::string>>("parallel_distribution")
    .SetPhysicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const ParallelDistribution& paral_dist =
          ctx->ParallelDistribution4ArgNameAndIndex("sequence", 0);
      const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
      int64_t device_batch_size = ctx->Attr<int64_t>("batch_size");
      FOR_RANGE(size_t, i, 0, paral_dist.sbp_parallel_size()) {
        const auto& sbp_parallel = paral_dist.sbp_parallel(i);
        if (sbp_parallel.has_split_parallel()) {
          int64_t split_num = hierarchy.At(i);
          CHECK_EQ_OR_RETURN(device_batch_size % split_num, 0);
          device_batch_size /= split_num;
        }
      }

      int64_t num_tokens = ctx->Attr<int64_t>("seq_length") + 1;
      user_op::TensorDesc* sequence_desc = ctx->TensorDesc4ArgNameAndIndex("sequence", 0);
      *sequence_desc->mut_shape() = Shape({device_batch_size, num_tokens});
      *sequence_desc->mut_data_type() = ctx->Attr<DataType>("dtype");
      return Maybe<void>::Ok();
    })
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      int64_t batch_size = ctx->Attr<int64_t>("batch_size");
      int64_t num_tokens = ctx->Attr<int64_t>("seq_length") + 1;
      user_op::TensorDesc* sequence_desc = ctx->TensorDesc4ArgNameAndIndex("sequence", 0);
      *sequence_desc->mut_shape() = Shape({batch_size, num_tokens});
      *sequence_desc->mut_data_type() = ctx->Attr<DataType>("dtype");
      return Maybe<void>::Ok();
    })
    .SetInferParallelDistributionFn([](user_op::InferParallelDistributionFnContext* ctx)
                                        -> Maybe<void> {
      const Shape& hierarchy = ctx->parallel_hierarchy();
      ParallelDistribution* seq_dist = ctx->ParallelDistribution4ArgNameAndIndex("sequence", 0);
      ParallelDistribution* iter_dist = ctx->ParallelDistribution4ArgNameAndIndex("iteration", 0);
      const auto& dist_conf =
          ctx->user_op_conf().attr<std::vector<std::string>>("parallel_distribution");
      CHECK_EQ_OR_RETURN(dist_conf.size(), hierarchy.NumAxes());
      for (const std::string& sbp_str : dist_conf) {
        SbpParallel sbp_parallel;
        CHECK_OR_RETURN(ParseSbpParallelFromString(sbp_str, &sbp_parallel));
        *seq_dist->add_sbp_parallel() = sbp_parallel;
        iter_dist->add_sbp_parallel()->mutable_broadcast_parallel();
      }
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](const user_op::GetInputArgModifier& GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper& conf) -> void {
      user_op::InputArgModifier* iteration_modifier = GetInputArgModifierFn("iteration", 0);
      CHECK(iteration_modifier != nullptr);
      iteration_modifier->set_is_mutable(true);
      iteration_modifier->set_requires_grad(false);
    })
    .SetOutputArgModifyFn([](user_op::GetOutputArgModifier GetOutputArgModifierFn,
                             const user_op::UserOpConfWrapper& conf) {
      user_op::OutputArgModifier* sequence_modifier = GetOutputArgModifierFn("sequence", 0);
      CHECK(sequence_modifier != nullptr);
      sequence_modifier->set_header_infered_before_compute(false);
    });

}  // namespace oneflow
