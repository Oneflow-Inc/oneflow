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

namespace {

Maybe<void> InferBlobDescs(const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  BlobDesc4BnInOp("out")->mut_shape() = Shape({1});
  return Maybe<void>::Ok();
}

}  // namespace

class DstSubsetTickOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DstSubsetTickOp);
  DstSubsetTickOp() = default;
  ~DstSubsetTickOp() = default;

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

void DstSubsetTickOp::InitFromOpConf() {
  CHECK(op_conf().has_dst_subset_tick_conf());
  EnrollRepeatedInputBn("in", false);
  EnrollOutputBn("out", false);
}

LogicalNode* DstSubsetTickOp::NewProperLogicalNode() const {
  return new DstSubsetTickLogicalNode();
}

Maybe<void> DstSubsetTickOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  return InferBlobDescs(BlobDesc4BnInOp);
}

Maybe<void> DstSubsetTickOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature*) const {
  return InferBlobDescs(GetBlobDesc4BnInOp);
}

Maybe<void> DstSubsetTickOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Broadcast(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kDstSubsetTickConf, DstSubsetTickOp);
REGISTER_TICK_TOCK_OP(OperatorConf::kDstSubsetTickConf);

}  // namespace oneflow
