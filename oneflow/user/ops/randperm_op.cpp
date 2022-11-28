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
#include "oneflow/core/job/nd_sbp_util.h"

namespace oneflow {

/*static*/ Maybe<void> RandpermOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  SbpParallel default_sbp;
  default_sbp.mutable_broadcast_parallel();
  return user_op::InferNdSbp4SrcOp(ctx, default_sbp);
}
/*static*/ Maybe<void> RandpermOp::GetSbp(user_op::SbpContext* ctx) { return Maybe<void>::Ok(); }
/*static*/ Maybe<void> RandpermOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  int32_t n = ctx->Attr<int32_t>("n");
  CHECK_GE_OR_RETURN(n, 0) << Error::RuntimeError()
                           << "Trying to create tensor with negative dimension " << n << ":"
                           << " [" << n << "]";
  ctx->SetOutputShape("out", 0, Shape({n}));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> RandpermOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& parallel_hierarchy = *ctx->parallel_desc().hierarchy();
  const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  int32_t n = ctx->Attr<int32_t>("n");
  const Shape& logical_shape = Shape({n});
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  const auto tensor_slice_view =
      GetTensorSliceView4ParallelId(parallel_hierarchy, nd_sbp, logical_shape, parallel_id);
  const Shape& physical_shape = tensor_slice_view.shape();

  ctx->SetOutputShape("out", 0, physical_shape);

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> RandpermOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, DataType::kInt32);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
