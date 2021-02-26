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

namespace {

Maybe<void> InferBlobDescs(const OperatorConf& op_conf,
                           const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  const ConstantLikeOpConf& conf = op_conf.constant_like_conf();
  BlobDesc* out_blob_desc = BlobDesc4BnInOp("out");
  *out_blob_desc = *BlobDesc4BnInOp("like");
  if (conf.has_data_type()) { out_blob_desc->set_data_type(conf.data_type()); }
  return Maybe<void>::Ok();
}

}  // namespace

class ConstantLikeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstantLikeOp);
  ConstantLikeOp() = default;
  ~ConstantLikeOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_constant_like_conf());
    EnrollInputBn("like", false);
    EnrollOutputBn("out", false);
  }

  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return InferBlobDescs(op_conf(), BlobDesc4BnInOp);
  }

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override {
    return InferBlobDescs(op_conf(), GetBlobDesc4BnInOp);
  }

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const {
    SbpSignatureBuilder()
        .Split("like", 0)
        .Split("out", 0)
        .MakeSplitSignatureListBuilder(JUST(LogicalBlobDesc4Ibn("like")).shape().NumAxes())
        .Build(sbp_sig_list);
    SbpSignatureBuilder().PartialSum("like").Broadcast("out").Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kConstantLikeConf, ConstantLikeOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kConstantLikeConf, 1);

}  // namespace oneflow
