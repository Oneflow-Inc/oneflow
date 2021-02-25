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
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class SrcSubsetTickOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SrcSubsetTickOp);
  SrcSubsetTickOp() = default;
  ~SrcSubsetTickOp() = default;

  void InitFromOpConf() override;
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature*) const override;
  LogicalNode* NewProperLogicalNode() const override;

 private:
  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
};

void SrcSubsetTickOp::InitFromOpConf() {
  CHECK(op_conf().has_src_subset_tick_conf());
  EnrollRepeatedInputBn("in", false);
  EnrollOutputBn("out", false);
}

LogicalNode* SrcSubsetTickOp::NewProperLogicalNode() const {
  return new SrcSubsetTickLogicalNode();
}

namespace {

Maybe<void> InferBlobDescs(const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  BlobDesc4BnInOp("out")->mut_shape() = Shape({1});
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> SrcSubsetTickOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  return InferBlobDescs(BlobDesc4BnInOp);
}

Maybe<void> SrcSubsetTickOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature*) const {
  return InferBlobDescs(GetBlobDesc4BnInOp);
}

Maybe<void> SrcSubsetTickOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Broadcast(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kSrcSubsetTickConf, SrcSubsetTickOp);
REGISTER_TICK_TOCK_OP(OperatorConf::kSrcSubsetTickConf);

}  // namespace oneflow
