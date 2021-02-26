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

Maybe<void> InferBlobDescs(const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  const BlobDesc* in_desc = BlobDesc4BnInOp("in");
  CHECK_OR_RETURN(in_desc->is_tensor_list());
  const int64_t N = in_desc->shape().At(0);
  BlobDesc* out_desc = BlobDesc4BnInOp("out");
  out_desc->mut_shape() = Shape({N});
  out_desc->set_data_type(DataType::kTensorBuffer);
  out_desc->set_is_dynamic(in_desc->is_dynamic());
  return Maybe<void>::Ok();
}

}  // namespace

class TensorListToTensorBufferOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorListToTensorBufferOp);
  TensorListToTensorBufferOp() = default;
  ~TensorListToTensorBufferOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_tensor_list_to_tensor_buffer_conf());
    EnrollInputBn("in", false);
    EnrollOutputBn("out", false)->set_header_infered_before_compute(false);
  }

  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return InferBlobDescs(BlobDesc4BnInOp);
  }

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override {
    return InferBlobDescs(GetBlobDesc4BnInOp);
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

REGISTER_CPU_OP(OperatorConf::kTensorListToTensorBufferConf, TensorListToTensorBufferOp);

}  // namespace oneflow
