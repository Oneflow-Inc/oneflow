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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> AccCtrlTickOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, Shape({1}));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> AccCtrlTickOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> AccCtrlTickOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> AccCtrlTickOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  const NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel_size(),  // NOLINT(maybe-need-error-msg)
                     parallel_hierarchy.NumAxes());    // NOLINT(maybe-need-error-msg)

  NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
  in_distribution->clear_sbp_parallel();
  out_distribution->clear_sbp_parallel();
  // in use hint
  in_distribution->CopyFrom(in_dis_hint);

  for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
    // out dim1 = broadcast
    out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AccCtrlTickOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> AccCtrlTickOp::InferOutputBlobTimeShape(
    user_op::InferOutputBlobTimeShapeFnContext* ctx) {
  const int32_t max_acc_num = ctx->user_op_conf().attr<int32_t>("max_acc_num");
  const Shape& in_time_shape = ctx->TimeShape4InputArgNameAndIndex("in", 0);
  DimVector time_shape_dim_vec = in_time_shape.dim_vec();  // NOLINT(maybe-need-error-msg)
  CHECK_OR_RETURN(!time_shape_dim_vec.empty());            // NOLINT(maybe-need-error-msg)
  if (time_shape_dim_vec.back() == max_acc_num) {
    time_shape_dim_vec.pop_back();
  } else if (time_shape_dim_vec.back() % max_acc_num == 0) {
    time_shape_dim_vec.back() /= max_acc_num;
  } else {
    const int64_t elem_cnt = in_time_shape.elem_cnt();
    CHECK_EQ_OR_RETURN(elem_cnt % max_acc_num, 0);
    time_shape_dim_vec.resize(1);
    time_shape_dim_vec.back() = elem_cnt / max_acc_num;
  }
  *ctx->mut_output_blob_time_shape() = Shape(time_shape_dim_vec);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
