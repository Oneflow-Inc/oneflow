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
#include "oneflow/core/operator/esac_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void EsacOp::InitFromOpConf() {
  EnrollRepeatedInputBn("in", false);
  EnrollOutputBn("out", false);
}

namespace {

Maybe<void> InferBlobDescs(const OperatorConf& op_conf,
                           const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  BlobDesc* out = BlobDesc4BnInOp("out");
  out->mut_shape() = Shape({1});
  const DataType data_type = op_conf.esac_conf().data_type();
  CHECK_OR_RETURN(IsIntegralDataType(data_type));
  out->set_data_type(data_type);
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> EsacOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  return InferBlobDescs(op_conf(), BlobDesc4BnInOp);
}

Maybe<void> EsacOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  return InferBlobDescs(op_conf(), GetBlobDesc4BnInOp);
}

Maybe<void> EsacOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  return Maybe<void>::Ok();
}

LogicalNode* EsacOp::NewProperLogicalNode() const { return new EsacLogicalNode(); }

REGISTER_CPU_OP(OperatorConf::kEsacConf, EsacOp);

}  // namespace oneflow
