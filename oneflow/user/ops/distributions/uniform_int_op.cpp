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

/* static */ Maybe<void> UniformIntOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  Shape* out_shape = ctx->OutputShape("out", 0);
  const Shape& shape = ctx->Attr<Shape>("shape");
  DimVector dim_vec;
  if (shape.NumAxes() > 0) {
    dim_vec.insert(dim_vec.end(), shape.dim_vec().cbegin(), shape.dim_vec().cend());
  }
  *out_shape = Shape(dim_vec);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UniformIntOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
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

/* static */ Maybe<void> UniformIntOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> UniformIntOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  cfg::SbpParallel default_sbp;
  default_sbp.mutable_broadcast_parallel();
  return user_op::InferNdSbp4SrcOp(ctx, default_sbp);
}

/* static */ Maybe<void> UniformIntOp::InferDataType(user_op::InferContext* ctx) {
  auto dtype = ctx->Attr<DataType>("dtype");
  *ctx->OutputDType("out", 0) = dtype;
  return Maybe<void>::Ok();
}

}  // namespace oneflow
