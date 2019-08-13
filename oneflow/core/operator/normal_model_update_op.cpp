#include "oneflow/core/operator/naive_model_update_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void NormalModelUpdtOp::InitFromOpConf() {
  EnrollInputBn("model_diff", false);
  EnrollInputBn("total_instance_num_diff", false);
  EnrollInputBn("model", false)->set_is_mutable(true);
  const PbMessage& conf = this->GetCustomizedConf();
  const auto& user_conf = *GetMsgPtrFromPbMessage<NormalModelUpdateOpUserConf>(conf, "user_conf");
  if (user_conf.has_clip_conf() && user_conf.clip_conf().has_clip_by_global_norm()) {
    EnrollDataTmpBn("data_tmp");
  }
  MdUpdtVirtualInitFromOpConf();
}

void NormalModelUpdtOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const PbMessage& conf = this->GetCustomizedConf();
  const auto& user_conf = *GetMsgPtrFromPbMessage<NormalModelUpdateOpUserConf>(conf, "user_conf");
  if (user_conf.has_clip_conf() && user_conf.clip_conf().has_clip_by_global_norm()) {
    *GetBlobDesc4BnInOp("data_tmp") = *GetBlobDesc4BnInOp("model_diff");
    GetBlobDesc4BnInOp("data_tmp")->mut_shape() = Shape({1});
  }
  MdUpdtVirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

const PbMessage& NormalModelUpdtOp::GetCustomizedConf() const {
  return op_conf().normal_mdupdt_conf();
}

LogicalBlobId NormalModelUpdtOp::obn2lbi(const std::string& output_bn) const {
  const google::protobuf::Descriptor* desc = GetCustomizedConf().GetDescriptor();
  const google::protobuf::FieldDescriptor* fd = desc->FindFieldByName(output_bn);
  CHECK(fd);
  return GenLogicalBlobId(GetValFromCustomizedConf<std::string>(output_bn));
}

void NormalModelUpdtOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  for (const auto& ibn : input_bns()) { CHECK_EQ(*HasBatchDim4BnInOp(ibn), false); }
}

void NormalModelUpdtOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const auto& bns = AlwaysBroadcastParallelBns();
  PbRpf<std::string> broadcast_bns = {bns.begin(), bns.end()};
  *broadcast_bns.Add() = "total_instance_num_diff";
  FOR_RANGE(int64_t, i, 0, LogicalBlobDesc4Ibn("model").shape().NumAxes()) {
    SbpSignatureBuilder()
        .Split(input_bns(), i)
        .Broadcast(broadcast_bns)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
}

REGISTER_OP_CREATOR(OperatorConf::kNormalMdupdtConf, [](const OperatorConf& op_conf) -> Operator* {
  return NewObj<NormalModelUpdtOp>(op_conf.normal_mdupdt_conf().user_conf().normal_mdupdt_case());
});

}  // namespace oneflow
