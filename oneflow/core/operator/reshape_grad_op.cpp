#include "oneflow/core/operator/reshape_grad_op.h"

namespace oneflow {

void ReshapeGradOp::InitFromOpConf() {
  CHECK(op_conf().has_reshape_grad_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollInputBn("like")->set_use_header_only(true);
}

const PbMessage& ReshapeGradOp::GetCustomizedConf() const { return op_conf().reshape_grad_conf(); }

void ReshapeGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("like");
}

REGISTER_OP(OperatorConf::kReshapeGradConf, ReshapeGradOp);

}  // namespace oneflow
