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

Maybe<void> InferConstantNdSbp(user_op::InferNdSbpFnContext* ctx);

REGISTER_NO_GRAD_USER_OP("constant")
    .Output("out")
    .SetOutputBufferNum(1)
    .Attr<double>("floating_value")
    .Attr<int64_t>("integer_value")
    .Attr<bool>("is_floating_value")
    .Attr<DataType>("dtype")
    .Attr<Shape>("shape")
    .Attr<std::vector<std::string>>("parallel_distribution")
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
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      auto dtype = ctx->Attr<DataType>("dtype");
      *ctx->OutputDType("out", 0) = dtype;
      return Maybe<void>::Ok();
    })
    .SetNdSbpInferFn([](user_op::InferNdSbpFnContext* ctx)
                                        -> Maybe<void> {
      const Shape& hierarchy = ctx->parallel_hierarchy();
      cfg::NdSbp* output_dist = ctx->NdSbp4ArgNameAndIndex("out", 0);
      // the input may be produced by iteration variable or tick, and all of them should be
      // broadcast parallel dist
      std::vector<cfg::NdSbp*> inputs_dist;
      for (const auto& arg_pair : ctx->inputs()) {
        inputs_dist.emplace_back(
            ctx->NdSbp4ArgNameAndIndex(arg_pair.first, arg_pair.second));
      }
      const auto& dist_conf =
          ctx->user_op_conf().attr<std::vector<std::string>>("parallel_distribution");
      if (dist_conf.size() == 0) {
        FOR_RANGE(int, i, 0, hierarchy.NumAxes()) {
          output_dist->add_sbp_parallel()->mutable_broadcast_parallel();
          for (auto* input_dist : inputs_dist) {
            input_dist->add_sbp_parallel()->mutable_broadcast_parallel();
          }
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
          for (auto* input_dist : inputs_dist) {
            input_dist->add_sbp_parallel()->mutable_broadcast_parallel();
          }
        }
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
