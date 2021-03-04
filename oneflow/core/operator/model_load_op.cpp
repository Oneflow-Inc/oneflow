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

class ModelLoadOp : public Operator {
 public:
  void InitFromOpConf() override;

  virtual Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void ModelLoadOp::InitFromOpConf() {
  CHECK(op_conf().has_model_load_conf());
  EnrollInputBn("path", false);
  EnrollRepeatedOutputBn("out", false);
}

namespace {

Maybe<void> InferBlobDescs(const OperatorConf& op_conf,
                           const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  const int64_t num_out = op_conf.model_load_conf().out().size();
  FOR_RANGE(int64_t, i, 0, num_out) {
    const VariableOpConf& original_variable_conf =
        op_conf.model_load_conf().original_variable_conf(i);
    BlobDesc* out_i = BlobDesc4BnInOp(GenRepeatedBn("out", i));
    out_i->mut_shape() = Shape(original_variable_conf.shape());
    out_i->set_data_type(original_variable_conf.data_type());
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> ModelLoadOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  return InferBlobDescs(op_conf(), BlobDesc4BnInOp);
}

Maybe<void> ModelLoadOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  return InferBlobDescs(op_conf(), GetBlobDesc4BnInOp);
}

Maybe<void> ModelLoadOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(
          JUST(LogicalBlobDesc4Ibn(output_bns().Get(0))).shape().NumAxes())
      .Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kModelLoadConf, ModelLoadOp);

}  // namespace oneflow
