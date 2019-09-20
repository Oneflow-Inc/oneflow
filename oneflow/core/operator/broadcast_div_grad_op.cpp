#include "oneflow/core/operator/broadcast_div_grad_op.h"

namespace oneflow {

void BroadcastDivGradOp::InitFromOpConf() {
  CHECK(op_conf().has_broadcast_div_grad_conf());
  EnrollInputBn("b");
  EnrollInputBn("y");
  EnrollInputBn("dy");
  EnrollOutputBn("db");
  EnrollOutputBn("temp_storage");
}

const PbMessage& BroadcastDivGradOp::GetCustomizedConf() const {
  return op_conf().broadcast_div_grad_conf();
}

Maybe<void> BroadcastDivGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  *GetBlobDesc4BnInOp("db") = *GetBlobDesc4BnInOp("b");
  *GetBlobDesc4BnInOp("temp_storage") = *GetBlobDesc4BnInOp("y");
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBroadcastDivGradConf, BroadcastDivGradOp);

}  // namespace oneflow
