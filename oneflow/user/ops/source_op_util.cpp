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

#include "oneflow/user/ops/source_op_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace user_op {

Maybe<void> InferSourceOpPhysicalTensorDesc(InferContext* ctx) {
  const Shape& shape = ctx->Attr<Shape>("shape");
  DimVector dim_vec{shape.dim_vec()};
  const cfg::NdSbp nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);

  // NOTE(wyg): 1d sbp support balanced splits but nd sbp not.
  if (nd_sbp.sbp_parallel_size() == 1) {
    const cfg::SbpParallel& sbp = nd_sbp.sbp_parallel(0);
    if (sbp.has_split_parallel()) {
      const int64_t& parallel_num = ctx->parallel_ctx().parallel_num();
      if (parallel_num > 1) {
        const int64_t& split_axis = sbp.split_parallel().axis();
        CHECK_LT_OR_RETURN(split_axis, dim_vec.size());
        BalancedSplitter bs(shape.At(split_axis), parallel_num);
        dim_vec[split_axis] = bs.At(ctx->parallel_ctx().parallel_id()).size();
      }
    }
  } else {
    const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
    for (int32_t i = 0; i < nd_sbp.sbp_parallel_size(); ++i) {
      const cfg::SbpParallel& sbp = nd_sbp.sbp_parallel(i);
      if (sbp.has_split_parallel()) {
        const int64_t& split_axis = sbp.split_parallel().axis();
        CHECK_EQ_OR_RETURN(dim_vec.at(split_axis) % hierarchy.At(i), 0)
            << "tensor physical shape in each device must be same in nd_sbp";
        dim_vec.at(split_axis) /= hierarchy.At(i);
      }
    }
  }

  *ctx->OutputShape("out", 0) = Shape(dim_vec);
  return Maybe<void>::Ok();
}

}  // namespace user_op
}  // namespace oneflow
