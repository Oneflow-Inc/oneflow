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
#include "oneflow/core/job/nd_sbp_util.h"

namespace oneflow {

/* static */ Maybe<void> UniformOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& shape = ctx->Attr<Shape>("shape");
  DimVector dim_vec;
  if (shape.NumAxes() > 0) {
    dim_vec.insert(dim_vec.end(), shape.dim_vec().cbegin(), shape.dim_vec().cend());
  }
  ctx->SetOutputShape("out", 0, Shape(dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UniformOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& parallel_hierarchy = *ctx->parallel_desc().hierarchy();
  const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  const Shape& logical_shape = ctx->Attr<Shape>("shape");
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  const auto tensor_slice_view =
      GetTensorSliceView4ParallelId(parallel_hierarchy, nd_sbp, logical_shape, parallel_id);
  const Shape& physical_shape = tensor_slice_view.shape();

  ctx->SetOutputShape("out", 0, physical_shape);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> UniformOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& logical_shape = ctx->Attr<Shape>("shape");
  int64_t num_axes = logical_shape.NumAxes();
  for (int i = 0; i < num_axes; ++i) {
    ctx->NewBuilder().Broadcast(ctx->inputs()).Split(ctx->outputs(), i).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> UniformOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  SbpParallel default_sbp;
  default_sbp.mutable_broadcast_parallel();
  return user_op::InferNdSbp4SrcOp(ctx, default_sbp);
}

/* static */ Maybe<void> UniformOp::InferDataType(user_op::InferContext* ctx) {
  auto dtype = ctx->Attr<DataType>("dtype");
  ctx->SetOutputDType("out", 0, dtype);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> UniformOp::DumpNdSbpSignatureForOpConfFn(const NdSbpSignature& nd_sbp_sig,
                                                                  OperatorConf* op_conf) {
  return user_op::SetSrcOpNdSbp(nd_sbp_sig, "out_0", op_conf);
}

}  // namespace oneflow
