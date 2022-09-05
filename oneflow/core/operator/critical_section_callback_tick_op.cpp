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
  BlobDesc* blob_desc = BlobDesc4BnInOp("out");
  blob_desc->set_shape(Shape({1}));
  blob_desc->set_data_type(DataType::kInt8);
  return Maybe<void>::Ok();
}

}  // namespace

class CriticalSectionCallbackTickOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CriticalSectionCallbackTickOp);
  CriticalSectionCallbackTickOp() = default;
  ~CriticalSectionCallbackTickOp() = default;

  Maybe<void> InitFromOpConf() override;
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override;
  Maybe<void> InferOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

Maybe<void> CriticalSectionCallbackTickOp::InitFromOpConf() {
  CHECK(op_conf().has_critical_section_callback_tick_conf());
  EnrollRepeatedInputBn("tick", false);
  EnrollOutputBn("out", false);
  return Maybe<void>::Ok();
}

Maybe<void> CriticalSectionCallbackTickOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  return InferBlobDescs(BlobDesc4BnInOp);
}

Maybe<void> CriticalSectionCallbackTickOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  return InferBlobDescs(GetBlobDesc4BnInOp);
}

Maybe<void> CriticalSectionCallbackTickOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  return Maybe<void>::Ok();
}

REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kCriticalSectionCallbackTickConf, 128);
REGISTER_OP(OperatorConf::kCriticalSectionCallbackTickConf, CriticalSectionCallbackTickOp);
REGISTER_TICK_TOCK_OP(OperatorConf::kCriticalSectionCallbackTickConf);

}  // namespace oneflow
