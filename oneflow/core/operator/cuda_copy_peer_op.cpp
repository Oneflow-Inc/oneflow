#include "oneflow/core/operator/cuda_copy_peer_op.h"

namespace oneflow {

void CudaCopyPeerOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& CudaCopyPeerOp::GetCustomizedConf() const {
  return op_conf().cuda_copy_peer_conf();
}

REGISTER_OP(OperatorConf::kCudaCopyPeerConf, CudaCopyPeerOp);

}  // namespace oneflow
