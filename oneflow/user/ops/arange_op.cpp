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

/* static */ Maybe<void> ArangeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  DataType dtype = ctx->Attr<DataType>("dtype");
  int64_t range_elem_cnt = 0;
  if (IsIntegralDataType(dtype)) {
    int64_t integer_delta = ctx->Attr<int64_t>("integer_delta");
    CHECK_NE_OR_RETURN(integer_delta, static_cast<int64_t>(0))
        << "RuntimeError: step must be nonzero. ";
    int64_t integer_start = ctx->Attr<int64_t>("integer_start");
    int64_t integer_limit = ctx->Attr<int64_t>("integer_limit");
    // CHECK when limit > start, delta > 0; limit < start, delta < 0;
    CHECK_GE_OR_RETURN((integer_limit - integer_start) / integer_delta, static_cast<int64_t>(0))
        << "RuntimeError: upper bound and larger bound inconsistent with step sign";
    range_elem_cnt = std::ceil(static_cast<double>(integer_limit - integer_start) / integer_delta);
  } else {
    double float_delta = ctx->Attr<double>("float_delta");
    CHECK_NE_OR_RETURN(float_delta, static_cast<double>(0.0))
        << "RuntimeError: step must be nonzero. ";
    double float_start = ctx->Attr<double>("float_start");
    double float_limit = ctx->Attr<double>("float_limit");
    // CHECK when limit > start, delta > 0; limit < start, delta < 0;
    // CHECK_GE For 0-Dim Tensor
    CHECK_GE_OR_RETURN((float_limit - float_start) / float_delta, static_cast<double>(0.0))
        << "RuntimeError: upper bound and larger bound inconsistent with step sign";
    range_elem_cnt = std::ceil(static_cast<double>(float_limit - float_start) / float_delta);
  }
  ctx->SetOutputShape("out", 0, Shape({range_elem_cnt}));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ArangeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  DataType dtype = ctx->Attr<DataType>("dtype");
  int64_t range_elem_cnt = 0;
  if (IsIntegralDataType(dtype)) {
    int64_t integer_delta = ctx->Attr<int64_t>("integer_delta");
    if (integer_delta == static_cast<int64_t>(0)) {
      return Error::RuntimeError() << " step must be nonzero. ";
    }
    int64_t integer_start = ctx->Attr<int64_t>("integer_start");
    int64_t integer_limit = ctx->Attr<int64_t>("integer_limit");
    // CHECK when limit > start, delta > 0; limit < start, delta < 0;
    if ((integer_limit - integer_start) / integer_delta < static_cast<int64_t>(0)) {
      return Error::RuntimeError() << " upper bound and larger bound inconsistent with step sign";
    }
    range_elem_cnt = std::ceil(static_cast<double>(integer_limit - integer_start) / integer_delta);
  } else {
    double float_delta = ctx->Attr<double>("float_delta");
    if (float_delta == static_cast<double>(0.0)) {
      return Error::RuntimeError() << " step must be nonzero. ";
    }
    double float_start = ctx->Attr<double>("float_start");
    double float_limit = ctx->Attr<double>("float_limit");
    // CHECK when limit > start, delta > 0; limit < start, delta < 0;
    // CHECK_GE For 0-Dim Tensor
    if ((float_limit - float_start) / float_delta < static_cast<double>(0.0)) {
      return Error::RuntimeError() << " upper bound and larger bound inconsistent with step sign";
    }
    range_elem_cnt = std::ceil(static_cast<double>(float_limit - float_start) / float_delta);
  }
  const Shape& logical_shape = Shape({range_elem_cnt});
  const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  const Shape& parallel_hierarchy = *ctx->parallel_desc().hierarchy();

  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  const auto tensor_slice_view =
      GetTensorSliceView4ParallelId(parallel_hierarchy, nd_sbp, logical_shape, parallel_id);
  const Shape& physical_shape = tensor_slice_view.shape();

  ctx->SetOutputShape("out", 0, physical_shape);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ArangeOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ArangeOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  SbpParallel default_sbp;
  default_sbp.mutable_broadcast_parallel();
  return user_op::InferNdSbp4SrcOp(ctx, default_sbp);
}

/* static */ Maybe<void> ArangeOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->Attr<DataType>("dtype"));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
