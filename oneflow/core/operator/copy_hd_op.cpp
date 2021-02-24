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
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CopyHdOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdOp);
  CopyHdOp() = default;
  ~CopyHdOp() override = default;

  void InitFromOpConf() override;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override;

 private:
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    const SbpParallel& sbp_parallel = JUST(SbpInferHint4Ibn(input_bns().Get(0)))->sbp_parallel();
    (*bn2sbp)[input_bns().Get(0)] = sbp_parallel;
    (*bn2sbp)[output_bns().Get(0)] = sbp_parallel;
    return Maybe<void>::Ok();
  }
  LogicalBlobId lbi4ibn(const std::string& input_bn) const override;
  LogicalBlobId lbi4obn(const std::string& output_bn) const override;
};

void CopyHdOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

Maybe<void> CopyHdOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

LogicalBlobId CopyHdOp::lbi4ibn(const std::string& input_bn) const {
  if (this->op_conf().copy_hd_conf().has_lbi()) {
    return this->op_conf().copy_hd_conf().lbi();
  } else {
    return GenPackedLbi();
  }
}

LogicalBlobId CopyHdOp::lbi4obn(const std::string& output_bn) const {
  if (this->op_conf().copy_hd_conf().has_lbi()) {
    return this->op_conf().copy_hd_conf().lbi();
  } else {
    return GenPackedLbi();
  }
}

REGISTER_OP(OperatorConf::kCopyHdConf, CopyHdOp);

}  // namespace oneflow
