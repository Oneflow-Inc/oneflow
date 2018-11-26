#include "oneflow/core/operator/naive_model_update_op.h"
#include "oneflow/core/operator/rmsprop_model_update_op.h"
#include "oneflow/core/operator/momentum_model_update_op.h"
#include "oneflow/core/operator/lars_model_update_op.h"
#include "oneflow/core/operator/adam_model_update_op.h"

namespace oneflow {

void NormalModelUpdtOp::InitFromOpConf() {
  EnrollInputBn("model_diff", false);
  EnrollInputBn("total_instance_num_diff", false);
  EnrollOutputBn("model", false);
  if (op_conf().normal_mdupdt_conf().user_conf().has_clip_conf()
      && op_conf().normal_mdupdt_conf().user_conf().clip_conf().has_clip_by_global_norm()) {
    EnrollDataTmpBn("global_norm");
  }
  MdUpdtVirtualInitFromOpConf();
}

void NormalModelUpdtOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  if (op_conf().normal_mdupdt_conf().user_conf().has_clip_conf()
      && op_conf().normal_mdupdt_conf().user_conf().clip_conf().has_clip_by_global_norm()) {
    *GetBlobDesc4BnInOp("global_norm") = *GetBlobDesc4BnInOp("model_diff");
    GetBlobDesc4BnInOp("global_norm")->mut_shape() = Shape({1});
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

REGISTER_OP_CREATOR(OperatorConf::kNormalMdupdtConf, [](const OperatorConf& op_conf) -> Operator* {
  return NewObj<NormalModelUpdtOp>(op_conf.normal_mdupdt_conf().user_conf().normal_mdupdt_case());
});

}  // namespace oneflow
