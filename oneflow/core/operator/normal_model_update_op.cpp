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
#include "oneflow/core/operator/normal_model_update_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void NormalModelUpdtOp::InitFromOpConf() {
  EnrollInputBn("model_diff", false);
  EnrollInputBn("model", false)->set_is_mutable(true);
  EnrollInputBn("learning_rate", false);
  EnrollInputBn("train_step", false);
  MdUpdtVirtualInitFromOpConf();
}

Maybe<void> NormalModelUpdtOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  return MdUpdtVirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

LogicalBlobId NormalModelUpdtOp::lbi4obn(const std::string& output_bn) const {
  const google::protobuf::Descriptor* desc = GetCustomizedConf().GetDescriptor();
  const google::protobuf::FieldDescriptor* fd = desc->FindFieldByName(output_bn);
  CHECK(fd);
  return GenLogicalBlobId(GetValFromCustomizedConf<std::string>(output_bn));
}

Maybe<void> NormalModelUpdtOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  return Maybe<void>::Ok();
}

Maybe<void> NormalModelUpdtOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const auto& bns = AlwaysBroadcastParallelBns();
  PbRpf<std::string> broadcast_bns = {bns.begin(), bns.end()};
  *broadcast_bns.Add() = "learning_rate";
  *broadcast_bns.Add() = "train_step";
  FOR_RANGE(int64_t, i, 0, JUST(LogicalBlobDesc4Ibn("model")).shape().NumAxes()) {
    SbpSignatureBuilder()
        .Split(input_bns(), i)
        .Broadcast(broadcast_bns)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_CREATOR(OperatorConf::kNormalMdupdtConf, ([](const OperatorConf& op_conf) -> Operator* {
                      return NewObj<int32_t, NormalModelUpdtOp>(
                          op_conf.normal_mdupdt_conf().user_conf().normal_mdupdt_case());
                    }));

}  // namespace oneflow
