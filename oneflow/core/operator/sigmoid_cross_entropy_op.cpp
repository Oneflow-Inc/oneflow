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
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

class SigmoidCrossEntropyOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SigmoidCrossEntropyOp);
  SigmoidCrossEntropyOp() = default;
  ~SigmoidCrossEntropyOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_sigmoid_cross_entropy_conf());
    EnrollInputBn("prediction");
    EnrollInputBn("label", false);
    EnrollOutputBn("loss");
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    CHECK_EQ_OR_RETURN(op_conf().sigmoid_cross_entropy_conf().label_type(),
                       GetBlobDesc4BnInOp("label")->data_type());
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("prediction")->shape(),
                       GetBlobDesc4BnInOp("label")->shape());
    *GetBlobDesc4BnInOp("loss") = *GetBlobDesc4BnInOp("prediction");
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .MakeSplitSignatureListBuilder(
            JUST(LogicalBlobDesc4Ibn(input_bns().Get(0))).shape().NumAxes())
        .Build(sbp_sig_list);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kSigmoidCrossEntropyConf, SigmoidCrossEntropyOp);

class SigmoidCrossEntropyGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SigmoidCrossEntropyGradOp);
  SigmoidCrossEntropyGradOp() = default;
  ~SigmoidCrossEntropyGradOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_sigmoid_cross_entropy_grad_conf());
    EnrollInputBn("prediction");
    EnrollInputBn("loss_diff");
    EnrollInputBn("label", false);
    EnrollOutputBn("prediction_diff");
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    CHECK_EQ_OR_RETURN(op_conf().sigmoid_cross_entropy_grad_conf().label_type(),
                       GetBlobDesc4BnInOp("label")->data_type());
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("prediction")->shape(),
                       GetBlobDesc4BnInOp("label")->shape());
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("prediction")->shape(),
                       GetBlobDesc4BnInOp("loss_diff")->shape());
    *GetBlobDesc4BnInOp("prediction_diff") = *GetBlobDesc4BnInOp("prediction");
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .MakeSplitSignatureListBuilder(
            JUST(LogicalBlobDesc4Ibn(input_bns().Get(0))).shape().NumAxes())
        .Build(sbp_sig_list);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kSigmoidCrossEntropyGradConf, SigmoidCrossEntropyGradOp);

}  // namespace oneflow
