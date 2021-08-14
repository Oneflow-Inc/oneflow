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
#include "oneflow/core/job/sbp_parallel.h" // cfg::ParallelDistribution
#include "oneflow/core/framework/infer_nd_sbp_fn_context.h"

#include "oneflow/core/job/lazy_mode.h"

namespace oneflow {
namespace user_op {

Maybe<void> InferSourceOpParallelDistribution(user_op::InferParallelDistributionFnContext* ctx) {
  const auto& dist_conf = ctx->user_op_conf().attr<std::vector<std::string>>("nd_sbp");
  const Shape& hierarchy = ctx->parallel_hierarchy();

  // the input may be produced by iteration variable or tick, and all of them should be
  // broadcast parallel dist
  for (const auto& arg_pair : ctx->inputs()) {
    cfg::ParallelDistribution* input_dist =
        ctx->ParallelDistribution4ArgNameAndIndex(arg_pair.first, arg_pair.second);
    FOR_RANGE(int, i, 0, hierarchy.NumAxes()) {
      input_dist->add_sbp_parallel()->mutable_broadcast_parallel();
    }
  }

  for (const auto& arg_pair : ctx->outputs()) {
    cfg::ParallelDistribution* output_dist =
        ctx->ParallelDistribution4ArgNameAndIndex(arg_pair.first, arg_pair.second);
    if (dist_conf.size() == 0) {
      // the default parallel dist of outputs of dataset op should be split(0)
      FOR_RANGE(int, i, 0, hierarchy.NumAxes()) {
        output_dist->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
      }
    } else {
      CHECK_EQ_OR_RETURN(dist_conf.size(), hierarchy.NumAxes());
      for (const std::string& sbp_str : dist_conf) {
        cfg::SbpParallel sbp_parallel;
        CHECK_OR_RETURN(ParseSbpParallelFromString(sbp_str, &sbp_parallel));
        CHECK_OR_RETURN(
            (sbp_parallel.has_split_parallel() && sbp_parallel.split_parallel().axis() == 0)
            || sbp_parallel.has_broadcast_parallel());
        *output_dist->add_sbp_parallel() = sbp_parallel;
      }
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace user_op
}  // namespace oneflow
