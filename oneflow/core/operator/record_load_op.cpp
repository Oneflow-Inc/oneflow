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
#include "oneflow/core/operator/record_load_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void RecordLoadOp::InitFromOpConf() {
  CHECK(op_conf().has_record_load_conf());
  if (op_conf().record_load_conf().has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("out", false);
}

Maybe<void> RecordLoadOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  int64_t batch_size = op_conf().record_load_conf().batch_size();
  CHECK_GE_OR_RETURN(batch_size, parallel_ctx->parallel_num());
  CHECK_EQ_OR_RETURN(batch_size % parallel_ctx->parallel_num(), 0);
  out_blob_desc->mut_shape() = Shape({batch_size / parallel_ctx->parallel_num()});
  out_blob_desc->set_data_type(kOFRecord);
  return Maybe<void>::Ok();
}

void RecordLoadOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  int64_t device_piece_size = GetBlobDesc4BnInOp("out")->shape().At(0);
  kernel_conf->mutable_record_load_conf()->set_device_piece_size(device_piece_size);
  kernel_conf->mutable_record_load_conf()->set_parallel_id(parallel_ctx->parallel_id());
  kernel_conf->mutable_record_load_conf()->set_parallel_num(parallel_ctx->parallel_num());
}

Maybe<void> RecordLoadOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kRecordLoadConf, RecordLoadOp);

}  // namespace oneflow
