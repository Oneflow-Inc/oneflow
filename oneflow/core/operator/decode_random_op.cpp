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
#include "oneflow/core/operator/decode_random_op.h"

namespace oneflow {

void DecodeRandomOp::InitFromOpConf() {
  CHECK(op_conf().has_decode_random_conf());
  if (op_conf().decode_random_conf().has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("out", false);
}

void DecodeRandomOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->mutable_decode_random_conf()->set_random_seed(NewRandomSeed());
}

Maybe<void> DecodeRandomOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const DecodeRandomOpConf& conf = op_conf().decode_random_conf();
  DimVector dim_vec(1 + conf.shape().dim_size());
  int64_t batch_size = conf.batch_size();
  CHECK_GE_OR_RETURN(batch_size, parallel_ctx->parallel_num());
  CHECK_EQ_OR_RETURN(batch_size % parallel_ctx->parallel_num(), 0);
  dim_vec[0] = batch_size / parallel_ctx->parallel_num();
  FOR_RANGE(size_t, j, 1, dim_vec.size()) { dim_vec[j] = conf.shape().dim(j - 1); }
  out_blob_desc->mut_shape() = Shape(dim_vec);
  out_blob_desc->set_data_type(conf.data_type());
  return Maybe<void>::Ok();
}

Maybe<void> DecodeRandomOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kDecodeRandomConf, DecodeRandomOp);

}  // namespace oneflow
