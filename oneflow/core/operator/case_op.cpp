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
#include "oneflow/core/operator/case_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void CaseOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollRepeatedOutputBn("out", false);
}

Maybe<void> CaseOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_EQ_OR_RETURN(in->shape().elem_cnt(), 1);
  const DataType data_type = in->data_type();
  CHECK_OR_RETURN(IsIntegralDataType(data_type));
  for (const std::string& obn : output_bns()) {
    BlobDesc* out = GetBlobDesc4BnInOp(obn);
    out->mut_shape() = Shape({1});
    out->set_data_type(data_type);
  }
  return Maybe<void>::Ok();
}

Maybe<void> CaseOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const std::string& obn : output_bns()) { *BatchAxis4BnInOp(obn) = *BatchAxis4BnInOp("in"); }
  return Maybe<void>::Ok();
}

Maybe<void> CaseOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  return Maybe<void>::Ok();
}

LogicalNode* CaseOp::NewProperLogicalNode() const { return new CaseLogicalNode(); }

REGISTER_CPU_OP(OperatorConf::kCaseConf, CaseOp);

}  // namespace oneflow
