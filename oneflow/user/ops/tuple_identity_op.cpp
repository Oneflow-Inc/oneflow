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

/*static*/ Maybe<void> TupleIdentityOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> TupleIdentityOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const int64_t in_size = ctx->input_size("in");
  CHECK_EQ_OR_RETURN(ctx->output_size("out"), in_size);
  for (int64_t i = 0; i < in_size; ++i) {
    ctx->SetOutputShape("out", i, ctx->InputShape("in", i));
    ctx->SetIsDynamic4ArgNameAndIndex("out", i, ctx->InputIsDynamic("in", i));
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TupleIdentityOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TupleIdentityOp::InferDataType(user_op::InferContext* ctx) {
  const int64_t in_size = ctx->input_size("in");
  CHECK_EQ_OR_RETURN(ctx->output_size("out"), in_size);
  for (int64_t i = 0; i < in_size; ++i) { ctx->SetOutputDType("out", i, ctx->InputDType("in", i)); }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TupleIdentityOp::InferSbpSignature(
    user_op::InferSbpSignatureFnContext* ctx) {
  SbpSignature* signature = ctx->mutable_sbp_signature();
  const SbpSignature& sbp_signature_conf = ctx->sbp_signature_conf();
  auto* bn2sbp = signature->mutable_bn_in_op2sbp_parallel();
  const auto& bn2conf_sbp = sbp_signature_conf.bn_in_op2sbp_parallel();
  const int64_t in_size = ctx->user_op_conf().input_size("in");
  CHECK_EQ_OR_RETURN(ctx->user_op_conf().output_size("out"), in_size);
  for (int64_t i = 0; i < in_size; ++i) {
    const SbpParallel* sbp_parallel = nullptr;
    const std::string ibn = GenRepeatedBn("in", i);
    const std::string& obn = GenRepeatedBn("out", i);
    const auto& conf_sbp_it = bn2conf_sbp.find(obn);
    if (conf_sbp_it == bn2conf_sbp.end()) {
      sbp_parallel = &ctx->SbpParallelHint4InputArgNameAndIndex("in", i);
    } else {
      sbp_parallel = &conf_sbp_it->second;
    }
    (*bn2sbp)[ibn] = *sbp_parallel;
    (*bn2sbp)[obn] = *sbp_parallel;
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TupleIdentityOp::CheckAttr(const user_op::UserOpDefWrapper&,
                                                  const user_op::UserOpConfWrapper& op_conf) {
  CHECK_OR_RETURN(op_conf.input_size("in") >= 1);
  CHECK_OR_RETURN(op_conf.output_size("out") >= 1);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
