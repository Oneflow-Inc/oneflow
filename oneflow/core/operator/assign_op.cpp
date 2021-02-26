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

class AssignOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AssignOp);
  AssignOp() = default;
  ~AssignOp() override = default;

  void InitFromOpConf() override;
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override;

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void AssignOp::InitFromOpConf() {
  CHECK(op_conf().has_assign_conf());
  EnrollInputBn("ref")->set_is_mutable(true);
  EnrollInputBn("value");
}

std::string DebugString(const BlobDesc& blob_desc) {
  BlobDescProto blob_desc_proto;
  blob_desc.ToProto(&blob_desc_proto);
  return blob_desc_proto.DebugString();
}

namespace {

Maybe<void> InferBlobDescs(const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  CHECK_OR_RETURN(*BlobDesc4BnInOp("ref") == *BlobDesc4BnInOp("value"))
      << "\nref_blob_desc: " << DebugString(*BlobDesc4BnInOp("ref"))
      << "\nvalue_blob_desc: " << DebugString(*BlobDesc4BnInOp("value"));
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> AssignOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  return InferBlobDescs(BlobDesc4BnInOp);
}

Maybe<void> AssignOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  return InferBlobDescs(GetBlobDesc4BnInOp);
}

Maybe<void> AssignOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .MakeSplitSignatureListBuilder(
          JUST(LogicalBlobDesc4Ibn(input_bns().Get(0))).shape().NumAxes())
      .Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kAssignConf, AssignOp);

}  // namespace oneflow
