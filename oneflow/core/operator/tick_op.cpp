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
#include "oneflow/core/operator/tick_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void TickOp::InitFromOpConf() {
  EnrollRepeatedInputBn("tick", false);
  EnrollOutputBn("out", false);
}

Maybe<void> TickOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  GetBlobDesc4BnInOp("out")->mut_shape() = Shape({1});
  return Maybe<void>::Ok();
}

Maybe<void> TickOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  BatchAxis4BnInOp("out")->clear_value();
  return Maybe<void>::Ok();
}

Maybe<void> TickOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  return Maybe<void>::Ok();
}

REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kTickConf, 2);
REGISTER_OP(OperatorConf::kTickConf, TickOp);
REGISTER_TICK_TOCK_OP(OperatorConf::kTickConf);

class DelayTickOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DelayTickOp);
  DelayTickOp() = default;
  ~DelayTickOp() = default;

  void InitFromOpConf() override {
    EnrollInputBn("tick", false);
    EnrollOutputBn("out", false);
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    GetBlobDesc4BnInOp("out")->mut_shape() = Shape({1});
    return Maybe<void>::Ok();
  }

  LogicalNode* NewProperLogicalNode() const override { return new DelayTickLogicalNode; }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    BatchAxis4BnInOp("out")->clear_value();
    return Maybe<void>::Ok();
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .MakeSplitSignatureListBuilder(
            JUST(LogicalBlobDesc4Ibn(input_bns().Get(0))).shape().NumAxes())
        .Build(sbp_sig_list);
    SbpSignatureBuilder()
        .PartialSum(input_bns())
        .PartialSum(output_bns())
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
};

REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kDelayTickConf, 1);
REGISTER_OP(OperatorConf::kDelayTickConf, DelayTickOp);
REGISTER_TICK_TOCK_OP(OperatorConf::kDelayTickConf);

}  // namespace oneflow
