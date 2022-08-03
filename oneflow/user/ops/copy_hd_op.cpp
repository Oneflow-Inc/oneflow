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
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

Maybe<void> InferLogicalOutBlobDescs(user_op::InferContext* ctx) { UNIMPLEMENTED_THEN_RETURN(); }

Maybe<void> InferOutBlobDescs(user_op::InferContext* ctx) {
  *ctx->MutOutputTensorDesc("out", 0) = ctx->InputTensorDesc("in", 0);
  return Maybe<void>::Ok();
}

Maybe<void> InferSbpSignature(user_op::InferSbpSignatureFnContext* ctx) {
  auto* bn2sbp = ctx->mutable_sbp_signature()->mutable_bn_in_op2sbp_parallel();
  const std::string& ibn = GenRepeatedBn("in", 0);
  const std::string& obn = GenRepeatedBn("out", 0);
  const auto& sbp_parallel = ctx->SbpParallelHint4InputArgNameAndIndex("in", 0);
  (*bn2sbp)[ibn] = sbp_parallel;
  (*bn2sbp)[obn] = sbp_parallel;
  return Maybe<void>::Ok();
}

Maybe<void> FwGetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  // only for nchw
  FOR_RANGE(int64_t, i, 0, std::min(2, (int)tensor.shape().NumAxes())) {
    ctx->NewBuilder().Split(user_op::OpArg("in", 0), i).Split(user_op::OpArg("out", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferFWDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> CopyD2HOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalOutBlobDescs(ctx);
}

Maybe<void> CopyD2HOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferOutBlobDescs(ctx);
}

Maybe<void> CopyD2HOp::InferSbpSignature(user_op::InferSbpSignatureFnContext* ctx) {
  return InferSbpSignature(ctx);
}

Maybe<void> CopyD2HOp::GetSbp(user_op::SbpContext* ctx) { return FwGetSbpFn(ctx); }

Maybe<void> CopyD2HOp::InferDataType(user_op::InferContext* ctx) { return InferFWDataType(ctx); }

Maybe<void> CopyH2DOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalOutBlobDescs(ctx);
}

Maybe<void> CopyH2DOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferOutBlobDescs(ctx);
}

Maybe<void> CopyH2DOp::InferSbpSignature(user_op::InferSbpSignatureFnContext* ctx) {
  return InferSbpSignature(ctx);
}

Maybe<void> CopyH2DOp::GetSbp(user_op::SbpContext* ctx) { return FwGetSbpFn(ctx); }

Maybe<void> CopyH2DOp::InferDataType(user_op::InferContext* ctx) { return InferFWDataType(ctx); }

}  // namespace oneflow
