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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

/* static */ Maybe<void> EyeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  int64_t rows = ctx->Attr<int64_t>("rows");
  int64_t cols = ctx->Attr<int64_t>("cols");
  *ctx->OutputShape("out", 0) = Shape({rows, cols});
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EyeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& shape = ctx->Attr<Shape>("shape");
  DimVector dim_vec{shape.dim_vec()};

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

  *ctx->OutputShape("out", 0) = Shape(dim_vec);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EyeOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EyeOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->Attr<DataType>("dtype");
  return Maybe<void>::Ok();
}

}  // namespace oneflow
