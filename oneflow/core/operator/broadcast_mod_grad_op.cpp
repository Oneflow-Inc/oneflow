#include "oneflow/core/operator/broadcast_mod_grad_op.h"

namespace oneflow {

void BroadcastModGradOp::InitFromOpConf() {
  CHECK(op_conf().has_broadcast_mod_grad_conf());
  EnrollInputBn("b");
  EnrollInputBn("y");
  EnrollInputBn("dy");
  EnrollOutputBn("db");
  EnrollTmpBn("temp_storage");
}

const PbMessage& BroadcastModGradOp::GetCustomizedConf() const {
  return op_conf().broadcast_mod_grad_conf();
}

Maybe<void> BroadcastModGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  *GetBlobDesc4BnInOp("db") = *GetBlobDesc4BnInOp("b");
  *GetBlobDesc4BnInOp("temp_storage") = *GetBlobDesc4BnInOp("y");
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBroadcastModGradConf, BroadcastModGradOp);

}  // namespace oneflow
