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

class ModelSaveOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveOp);
  ModelSaveOp() = default;
  ~ModelSaveOp() override = default;

  void InitFromOpConf() override;
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return Maybe<void>::Ok();
  }
  Maybe<void> InferOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override {
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    return Maybe<void>::Ok();
  };

  Maybe<void> InferParallelDistributionSignature(
      ParallelDistributionSignature* parallel_distribution_signature,
      const ParallelDistributionSignature& parallel_distribution_constraints,
      const ParallelDesc& parallel_desc,
      std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
          ParallelDistributionInferHint4Ibn) const override {
    ParallelDistribution broadcast_distribution;
    for (int64_t i = 0; i < parallel_desc.hierarchy()->NumAxes(); ++i) {
      broadcast_distribution.add_sbp_parallel()->mutable_broadcast_parallel();
    }
    for (const std::string& ibn : input_bns()) {
      (*parallel_distribution_signature->mutable_bn_in_op2parallel_distribution())[ibn] =
          broadcast_distribution;
    }
    for (const std::string& obn : output_bns()) {
      (*parallel_distribution_signature->mutable_bn_in_op2parallel_distribution())[obn] =
          broadcast_distribution;
    }
    return Maybe<void>::Ok();
  }
};

void ModelSaveOp::InitFromOpConf() {
  CHECK(op_conf().has_model_save_conf());
  EnrollInputBn("path", false);
  EnrollRepeatedInputBn("in", false);
}

REGISTER_CPU_OP(OperatorConf::kModelSaveConf, ModelSaveOp);

}  // namespace oneflow
