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
/* static */ Maybe<void> OFRecordReaderOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  out_tensor->set_shape(Shape({ctx->Attr<int32_t>("batch_size")}));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OFRecordReaderOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  int32_t batch_size = ctx->Attr<int32_t>("batch_size");
  int64_t parallel_num = ctx->parallel_ctx().parallel_num();
  if (parallel_num > 1) {
    int64_t split_num = 1;
    const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
    const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
    for (int32_t i = 0; i < nd_sbp.sbp_parallel_size(); ++i) {
      if (nd_sbp.sbp_parallel(i).has_split_parallel()) { split_num *= hierarchy.At(i); }
    }
    CHECK_EQ_OR_RETURN(batch_size % split_num, 0);
    batch_size /= split_num;
  }
  out_tensor->set_shape(Shape({batch_size}));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OFRecordReaderOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OFRecordReaderOp::GetNdSbpSignatureList(
    user_op::GetNdSbpSignatureListContext* ctx) {
  NdSbpSignature nd_sbp_signature;
  SbpParallel split_sbp_parallel;
  split_sbp_parallel.mutable_split_parallel()->set_axis(0);
  for (int32_t dim_sbp = 0; dim_sbp < ctx->parallel_hierarchy().NumAxes(); dim_sbp++) {
    *(*nd_sbp_signature.mutable_bn_in_op2nd_sbp())[GenRepeatedBn("out", 0)].add_sbp_parallel() =
        split_sbp_parallel;
  }
  ctx->AddNdSbpSignature(nd_sbp_signature);
  return Maybe<void>::Ok();
}

/* static */ Maybe<double> OFRecordReaderOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  // Don't support broadcast.
  return double(ctx->Shape4ArgNameAndIndex("out", 0).elem_cnt()
                * GetSizeOfDataType(DataType::kOFRecord))
         / ctx->parallel_desc().hierarchy()->elem_cnt();
}

/* static */ Maybe<void> OFRecordReaderOp::ModifyOutputArg(
    const GetOutputArgModifier& GetOutputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::OutputArgModifier* out_modifier = GetOutputArgModifierFn("out", 0);
  CHECK_OR_RETURN(out_modifier != nullptr);
  // NOTE(chengcheng): OFRecordReader Only support static shape infer which will read all batch
  //  size data with output shape (batch_size,)
  // out_modifier->set_header_infered_before_compute(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OFRecordReaderOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  SbpParallel default_sbp;
  default_sbp.mutable_split_parallel()->set_axis(0);
  return user_op::InferNdSbp4SrcOp(ctx, default_sbp);
}

/* static */ Maybe<void> OFRecordReaderOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, DataType::kOFRecord);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
