#include "oneflow/core/operator/broadcast_div_op.h"

namespace oneflow {

void BroadcastDivOp::DerivedInitFromOpConf() { EnrollBwBufBn("bw_buf"); }

const PbMessage& BroadcastDivOp::GetCustomizedConf() const {
  return op_conf().broadcast_div_conf();
}

void BroadcastDivOp::DerivedInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("bw_buf") = *GetBlobDesc4BnInOp("out");
}

REGISTER_OP(OperatorConf::kBroadcastDivConf, BroadcastDivOp);

}  // namespace oneflow
