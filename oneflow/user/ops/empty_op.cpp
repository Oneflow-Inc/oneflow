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
#include "oneflow/core/common/global.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

REGISTER_NO_GRAD_USER_OP("empty")
    .Output("out")
    .SetOutputBufferNum(1)
    .Attr<DataType>("dtype")
    .Attr<Shape>("shape")
    .Attr<std::vector<std::string>>("nd_sbp")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->OutputShape("out", 0);
      const Shape& shape = ctx->Attr<Shape>("shape");
      DimVector dim_vec;
      if (shape.NumAxes() > 0) {
        dim_vec.insert(dim_vec.end(), shape.dim_vec().cbegin(), shape.dim_vec().cend());
      }
      *out_shape = Shape(dim_vec);
      return Maybe<void>::Ok();
    })
    .SetPhysicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->OutputShape("out", 0);
      const Shape& shape = ctx->Attr<Shape>("shape");
      DimVector dim_vec;
      if (shape.NumAxes() > 0) {
        dim_vec.insert(dim_vec.end(), shape.dim_vec().cbegin(), shape.dim_vec().cend());
      }

      const cfg::SbpParallel& out_sbp_para = ctx->SbpParallel4ArgNameAndIndex("out", 0);
      if (out_sbp_para.has_split_parallel()) {
        const int64_t& parallel_num = ctx->parallel_ctx().parallel_num();
        if (parallel_num > 1) {
          const int64_t& split_axis = out_sbp_para.split_parallel().axis();
          CHECK_LT_OR_RETURN(split_axis, dim_vec.size());
          BalancedSplitter bs(shape.At(split_axis), parallel_num);
          dim_vec[split_axis] = bs.At(ctx->parallel_ctx().parallel_id()).size();
        }
      }

      *out_shape = Shape(dim_vec);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const Shape& shape = ctx->Attr<Shape>("shape");
      if (shape.NumAxes() > 0) {
        FOR_RANGE(int64_t, i, 0, shape.NumAxes()) {
          ctx->NewBuilder().Split(ctx->outputs(), i).Build();
        }
      }
      ctx->NewBuilder().PartialSum(ctx->outputs()).Build();
      return Maybe<void>::Ok();
    })
    .SetSbpSignatureInferFn([](user_op::InferSbpSignatureFnContext* ctx) -> Maybe<void> {
      auto* bn2sbp = ctx->mutable_sbp_signature()->mutable_bn_in_op2sbp_parallel();
      const std::string& obn = GenRepeatedBn("out", 0);
      const auto& sbp_parallel_str = ctx->Attr<std::string>("sbp_parallel");
      const std::string& ibn = GenRepeatedBn(user_op::kUserSourceOpTickInputArgName, 0);
      cfg::SbpParallel sbp_parallel;
      sbp_parallel.mutable_broadcast_parallel();
      (*bn2sbp)[ibn] = sbp_parallel;
      if (sbp_parallel_str.empty()) {
        (*bn2sbp)[obn] = sbp_parallel;
      } else {
        CHECK_OR_RETURN(ParseSbpParallelFromString(sbp_parallel_str, &sbp_parallel))
            << "invalid sbp_parallel: " << sbp_parallel_str;
        if (sbp_parallel.has_split_parallel()) {
          int64_t split_axis = sbp_parallel.split_parallel().axis();
          const Shape& shape = ctx->Attr<Shape>("shape");
          CHECK_OR_RETURN(shape.NumAxes() > 0)
              << "Split parallel is not supported for shape whose value is None";
          CHECK_GE_OR_RETURN(split_axis, 0);
          CHECK_LT_OR_RETURN(split_axis, shape.NumAxes());
          (*bn2sbp)[obn] = sbp_parallel;
        } else if (sbp_parallel.has_broadcast_parallel()) {
          (*bn2sbp)[obn] = sbp_parallel;
        } else if (sbp_parallel.has_partial_sum_parallel()) {
          (*bn2sbp)[obn] = sbp_parallel;
        } else {
          UNIMPLEMENTED() << "sbp parallel not supported";
        }
      }
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const DataType dtype = ctx->Attr<DataType>("dtype");
      *ctx->OutputDType("out", 0) = dtype;
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn([](user_op::InferParallelDistributionFnContext* ctx)
                                        -> Maybe<void> {
      const Shape& hierarchy = ctx->parallel_hierarchy();
      cfg::ParallelDistribution* output_dist = ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
      // the input may be produced by iteration variable or tick, and all of them should be
      // broadcast parallel dist
      std::vector<cfg::ParallelDistribution*> inputs_dist;
      for (const auto& arg_pair : ctx->inputs()) {
        inputs_dist.emplace_back(
            ctx->ParallelDistribution4ArgNameAndIndex(arg_pair.first, arg_pair.second));
      }
      const auto& dist_conf = ctx->user_op_conf().attr<std::vector<std::string>>("nd_sbp");
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
