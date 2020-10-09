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

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const BlobDesc* in_desc = GetBlobDesc4BnInOp(SoleIbn());
    CHECK_OR_RETURN(in_desc->is_tensor_list());
    CHECK_GT_OR_RETURN(in_desc->shape().NumAxes(), 1);
    const int64_t N = in_desc->shape().At(0);
    CHECK_EQ_OR_RETURN(N, output_bns().size());
    DimVector dim_vec{in_desc->shape().dim_vec().begin() + 1, in_desc->shape().dim_vec().end()};
    FOR_RANGE(int, i, 0, N) {
      BlobDesc* out_i = GetBlobDesc4BnInOp(output_bns().Get(i));
      out_i->mut_shape() = Shape(dim_vec);
      out_i->set_data_type(in_desc->data_type());
      out_i->set_is_dynamic(true);
    }
    return Maybe<void>::Ok();
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

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    for (const auto& obn : output_bns()) { *BatchAxis4BnInOp(obn) = *BatchAxis4BnInOp(SoleIbn()); }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kTensorListSplitConf, TensorListSplitOp);

}  // namespace oneflow
