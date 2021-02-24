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
#include "oneflow/core/operator/decode_ofrecord_op.h"
#include "oneflow/core/record/ofrecord_decoder.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

int64_t GetDim0(int64_t batch_size, const ParallelContext& parallel_ctx) {
  BalancedSplitter bs(batch_size, parallel_ctx.parallel_num());
  return bs.At(parallel_ctx.parallel_id()).size();
}

}  // namespace

void DecodeOFRecordOp::InitFromOpConf() {
  CHECK(op_conf().has_decode_ofrecord_conf());
  if (op_conf().decode_ofrecord_conf().has_in()) { EnrollInputBn("in", false); }
  const DecodeOFRecordOpConf& conf = op_conf().decode_ofrecord_conf();
  for (int32_t i = 0; i < conf.blob_size(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
  if (conf.part_name_suffix_length() != -1) {
    CHECK_GE(conf.part_name_suffix_length(), std::to_string(conf.data_part_num() - 1).length());
  }
}

void DecodeOFRecordOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->mutable_decode_ofrecord_conf()->set_random_seed(NewRandomSeed());
}

Maybe<void> DecodeOFRecordOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  int64_t batch_size = op_conf().decode_ofrecord_conf().batch_size();
  int64_t dim0 = GetDim0(batch_size, *parallel_ctx);
  if (op_conf().decode_ofrecord_conf().has_in()) {
    BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
    CHECK_EQ(dim0, in_blob_desc->shape().At(0));
  }

  FOR_RANGE(size_t, i, 0, output_bns().size()) {
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
    const BlobConf& blob_conf = op_conf().decode_ofrecord_conf().blob(i);
    DimVector dim_vec(1 + blob_conf.shape().dim_size());
    dim_vec[0] = dim0;
    FOR_RANGE(size_t, j, 1, dim_vec.size()) { dim_vec[j] = blob_conf.shape().dim(j - 1); }
    out_blob_desc->mut_shape() = Shape(dim_vec);
    out_blob_desc->set_data_type(blob_conf.data_type());
  }
  return Maybe<void>::Ok();
}

LogicalBlobId DecodeOFRecordOp::lbi4obn(const std::string& output_bn) const {
  CHECK_STREQ(output_bn.substr(0, 4).c_str(), "out_");
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(
      op_conf().decode_ofrecord_conf().blob(oneflow_cast<int32_t>(output_bn.substr(4))).name());
  return ret;
}

Maybe<void> DecodeOFRecordOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kDecodeOfrecordConf, DecodeOFRecordOp);

}  // namespace oneflow
