#include "oneflow/core/operator/dropout_grad_op.h"

namespace oneflow {

void DropoutGradOp::InitFromOpConf() {
  double dropout_rate = op_conf().dropout_conf().rate();
  CHECK_GE(dropout_rate, 0);
  CHECK_LT(dropout_rate, 1);
  EnrollInputBn("dy");
  EnrollInputBn("random_mask");
  EnrollOutputBn("dx");
}

const PbMessage& DropoutGradOp::GetCustomizedConf() const { return op_conf().dropout_grad_conf(); }

void DropoutGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("dx") = *GetBlobDesc4BnInOp("dy");
}

REGISTER_OP(OperatorConf::kDropoutGradConf, DropoutGradOp);

}  // namespace oneflow
