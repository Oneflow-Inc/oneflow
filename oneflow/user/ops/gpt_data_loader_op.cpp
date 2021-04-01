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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {

REGISTER_CPU_ONLY_USER_OP("megatron_gpt_mmap_data_loader")
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
    .Attr<std::vector<std::string>>("parallel_distribution")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      int64_t batch_size = ctx->Attr<int64_t>("batch_size");
      int64_t sample_len = ctx->Attr<int64_t>("seq_length") + ctx->Attr<int64_t>("label_length");
      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc->mut_shape() = Shape({batch_size, sample_len});
      *out_desc->mut_data_type() = ctx->Attr<DataType>("dtype");
      return Maybe<void>::Ok();
    })
    .SetInferParallelDistributionFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const Shape& hierarchy = ctx->parallel_hierarchy();
          ParallelDistribution* output_dist = ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
          // the input may be produced by iteration variable or tick, and all of them should be
          // broadcast parallel dist
          std::vector<ParallelDistribution*> inputs_dist;
          for (const auto& arg_pair : ctx->inputs()) {
            inputs_dist.emplace_back(
                ctx->ParallelDistribution4ArgNameAndIndex(arg_pair.first, arg_pair.second));
          }
          const auto& dist_conf =
              ctx->user_op_conf().attr<std::vector<std::string>>("parallel_distribution");
          if (dist_conf.size() == 0) {
            FOR_RANGE(int, i, 0, hierarchy.NumAxes()) {
              output_dist->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
              for (auto* input_dist : inputs_dist) {
                input_dist->add_sbp_parallel()->mutable_broadcast_parallel();
              }
            }
          } else {
            CHECK_EQ_OR_RETURN(dist_conf.size(), hierarchy.NumAxes());
            for (const std::string& sbp_str : dist_conf) {
              SbpParallel sbp_parallel;
              CHECK_OR_RETURN(ParseSbpParallelFromString(sbp_str, &sbp_parallel));
              CHECK_OR_RETURN(
                  (sbp_parallel.has_split_parallel() && sbp_parallel.split_parallel().axis() == 0)
                  || sbp_parallel.has_broadcast_parallel());
              *output_dist->add_sbp_parallel() = sbp_parallel;
              for (auto* input_dist : inputs_dist) {
                input_dist->add_sbp_parallel()->mutable_broadcast_parallel();
              }
            }
          }
          return Maybe<void>::Ok();
        })
    .SetInputArgModifyFn([](const user_op::GetInputArgModifier& GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper& conf) -> void {
      if (!conf.has_input("iteration", 0)) { return; }
      user_op::InputArgModifier* input_modifier = GetInputArgModifierFn("iteration", 0);
      CHECK(input_modifier != nullptr);
      input_modifier->set_is_mutable(true);
      input_modifier->set_requires_grad(false);
    });

}  // namespace oneflow
