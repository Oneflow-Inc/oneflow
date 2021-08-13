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
      Shape* out_shape = ctx->OutputShape("out", 0);
      const Shape& shape = ctx->Attr<Shape>("shape");
      DimVector dim_vec;
      if (shape.NumAxes() > 0) {
        dim_vec.insert(dim_vec.end(), shape.dim_vec().cbegin(), shape.dim_vec().cend());
      }
      *out_shape = Shape(dim_vec);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      auto dtype = ctx->Attr<DataType>("dtype");
      *ctx->OutputDType("out", 0) = dtype;
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const Shape& hierarchy = ctx->parallel_hierarchy();
          const auto& dist_conf = ctx->user_op_conf().attr<std::vector<std::string>>("nd_sbp");

          // constant may have tick inputs whose sbp should be broadcast
          for (const auto& arg_pair : ctx->inputs()) {
            cfg::ParallelDistribution* input_dist =
                ctx->ParallelDistribution4ArgNameAndIndex(arg_pair.first, arg_pair.second);
            FOR_RANGE(int, i, 0, hierarchy.NumAxes()) {
              input_dist->add_sbp_parallel()->mutable_broadcast_parallel();
            }
          }

          const auto& outputs = ctx->outputs();
          CHECK_EQ_OR_RETURN(outputs.size(), 1);
          cfg::ParallelDistribution* output_dist =
              ctx->ParallelDistribution4ArgNameAndIndex(outputs[0].first, outputs[0].second);
          if (dist_conf.size() == 0) {
            // the default sbp of constant's output should be broadcast
            FOR_RANGE(int, i, 0, hierarchy.NumAxes()) {
              output_dist->add_sbp_parallel()->mutable_broadcast_parallel();
            }
          } else {
            CHECK_EQ_OR_RETURN(dist_conf.size(), hierarchy.NumAxes());
            for (const std::string& sbp_str : dist_conf) {
              cfg::SbpParallel sbp_parallel;
              CHECK_OR_RETURN(ParseSbpParallelFromString(sbp_str, &sbp_parallel));
              CHECK_OR_RETURN(sbp_parallel.has_split_parallel()
                              || sbp_parallel.has_broadcast_parallel());
              *output_dist->add_sbp_parallel() = sbp_parallel;
            }
          }

          return Maybe<void>::Ok();
        });

}  // namespace oneflow
