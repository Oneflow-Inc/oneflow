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
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {

Maybe<void> InferBlobDescs(const Operator& op,
                           const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  const BlobDesc* in_desc = BlobDesc4BnInOp(op.SoleIbn());
  CHECK_OR_RETURN(in_desc->is_tensor_list());
  CHECK_GT_OR_RETURN(in_desc->shape().NumAxes(), 1);
  const int64_t N = in_desc->shape().At(0);
  CHECK_EQ_OR_RETURN(N, op.output_bns().size());
  DimVector dim_vec{in_desc->shape().dim_vec().begin() + 1, in_desc->shape().dim_vec().end()};
  FOR_RANGE(int, i, 0, N) {
    BlobDesc* out_i = BlobDesc4BnInOp(op.output_bns().Get(i));
    out_i->mut_shape() = Shape(dim_vec);
    out_i->set_data_type(in_desc->data_type());
    out_i->set_is_dynamic(true);
  }
  return Maybe<void>::Ok();
}

}  // namespace

class TensorListSplitOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorListSplitOp);
  TensorListSplitOp() = default;
  ~TensorListSplitOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_tensor_list_split_conf());
    EnrollInputBn("in", false);
    EnrollRepeatedOutputBnWithSetter("out", false, [](OutputBlobModifier* ob_modifier) {
      ob_modifier->set_header_infered_before_compute(false);
    });
  }

  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return InferBlobDescs(*this, BlobDesc4BnInOp);
  }

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override {
    return InferBlobDescs(*this, GetBlobDesc4BnInOp);
  }

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kTensorListSplitConf, TensorListSplitOp);

}  // namespace oneflow
