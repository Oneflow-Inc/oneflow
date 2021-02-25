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

class LearningRateScheduleOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LearningRateScheduleOp);
  LearningRateScheduleOp() = default;
  ~LearningRateScheduleOp() override = default;

  void InitFromOpConf() override;
  virtual Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override;

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void LearningRateScheduleOp::InitFromOpConf() {
  CHECK(op_conf().has_learning_rate_schedule_conf());
  EnrollInputBn("train_step");
  EnrollOutputBn("out");
}

namespace {

Maybe<void> InferBlobDescs(const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  const BlobDesc* train_step = BlobDesc4BnInOp("train_step");
  CHECK_EQ(train_step->shape().elem_cnt(), 1);
  CHECK_EQ(train_step->data_type(), DataType::kInt64);
  BlobDesc* out = BlobDesc4BnInOp("out");
  out->mut_shape() = Shape({1});
  out->set_data_type(DataType::kFloat);
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> LearningRateScheduleOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  return InferBlobDescs(BlobDesc4BnInOp);
}

Maybe<void> LearningRateScheduleOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  return InferBlobDescs(GetBlobDesc4BnInOp);
}

Maybe<void> LearningRateScheduleOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kLearningRateScheduleConf, LearningRateScheduleOp);

}  // namespace oneflow
