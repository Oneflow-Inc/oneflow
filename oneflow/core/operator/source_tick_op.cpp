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
#include "oneflow/core/operator/source_tick_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

Maybe<void> SourceTickOp::InitFromOpConf() {
  CHECK(op_conf().has_source_tick_conf());
  CHECK(op_conf().ctrl_in_op_name().empty());
  if (op_conf().source_tick_conf().has_wait_in()) { EnrollInputBn("wait_in", false); }
  EnrollOutputBn("out", false);
  return Maybe<void>::Ok();
}

Maybe<void> SourceTickOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  BlobDesc* blob_desc = BlobDesc4BnInOp("out");
  blob_desc->mut_shape() = Shape({1});
  blob_desc->set_data_type(DataType::kUInt8);
  return Maybe<void>::Ok();
}

Maybe<void> SourceTickOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ_OR_RETURN(parallel_ctx->parallel_num(), 1);
  BlobDesc* blob_desc = GetBlobDesc4BnInOp("out");
  blob_desc->mut_shape() = Shape({1});
  blob_desc->set_data_type(DataType::kUInt8);
  return Maybe<void>::Ok();
}

Maybe<void> SourceTickOp::GetSbpSignatures(cfg::SbpSignatureList* sbp_sig_list) const {
  auto* sbp_signature = sbp_sig_list->mutable_sbp_signature()->Add();
  SbpSignatureBuilder().Broadcast(input_bns()).Broadcast(output_bns()).Build(sbp_signature);
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kSourceTickConf, SourceTickOp);
REGISTER_TICK_TOCK_OP(OperatorConf::kSourceTickConf);

}  // namespace oneflow
