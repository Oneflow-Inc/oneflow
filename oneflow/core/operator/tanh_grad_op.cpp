#include "oneflow/core/operator/tanh_grad_op.h"

namespace oneflow {

void TanHGradOp::InitFromOpConf() {
  CHECK(op_conf().has_tanh_grad_conf());
  EnrollInputBn("y");
  EnrollInputBn("dy");
  EnrollOutputBn("dx");
}

const PbMessage& TanHGradOp::GetCustomizedConf() const { return op_conf().tanh_grad_conf(); }

void TanHGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("dx") = *GetBlobDesc4BnInOp("y");
}

REGISTER_OP(OperatorConf::kTanhGradConf, TanHGradOp);

}  // namespace oneflow
