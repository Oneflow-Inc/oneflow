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
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

namespace {

Maybe<void> InferNdSbp4Constant(user_op::InferParallelDistributionFnContext* ctx) {
  const Shape& hierarchy = ctx->parallel_hierarchy();
  const auto& sbp_str_list = ctx->user_op_conf().attr<std::vector<std::string>>("nd_sbp");

  // constant may have tick inputs whose sbp should be broadcast
  for (const auto& arg_pair : ctx->inputs()) {
    cfg::ParallelDistribution* input_nd_sbp =
        ctx->ParallelDistribution4ArgNameAndIndex(arg_pair.first, arg_pair.second);
    FOR_RANGE(int, i, 0, hierarchy.NumAxes()) {
      input_nd_sbp->add_sbp_parallel()->mutable_broadcast_parallel();
    }
  }

  const auto& outputs = ctx->outputs();
  CHECK_EQ_OR_RETURN(outputs.size(), 1);
  cfg::ParallelDistribution* output_nd_sbp =
      ctx->ParallelDistribution4ArgNameAndIndex(outputs[0].first, outputs[0].second);
  if (sbp_str_list.size() == 0) {
    // the default sbp of constant's output should be broadcast
    FOR_RANGE(int, i, 0, hierarchy.NumAxes()) {
      output_nd_sbp->add_sbp_parallel()->mutable_broadcast_parallel();
    }
  } else {
    CHECK_EQ_OR_RETURN(sbp_str_list.size(), hierarchy.NumAxes());
    for (const std::string& sbp_str : sbp_str_list) {
      cfg::SbpParallel sbp;
      CHECK_OR_RETURN(ParseSbpParallelFromString(sbp_str, &sbp));
      CHECK_OR_RETURN(sbp.has_split_parallel() || sbp.has_broadcast_parallel());
      *output_nd_sbp->add_sbp_parallel() = sbp;
    }
  }

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_NO_GRAD_USER_OP("constant")
    .Output("out")
    .SetOutputBufferNum(1)
    .Attr<double>("floating_value")
    .Attr<int64_t>("integer_value")
    .Attr<bool>("is_floating_value")
    .Attr<DataType>("dtype")
    .Attr<Shape>("shape")
    .Attr<std::vector<std::string>>("nd_sbp")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = Shape(ctx->Attr<Shape>("shape").dim_vec());
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->Attr<DataType>("dtype");
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> { return Maybe<void>::Ok(); })
    .SetParallelDistributionInferFn(InferNdSbp4Constant);

}  // namespace oneflow
