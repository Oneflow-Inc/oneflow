#include "oneflow/core/operator/gelu_grad_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void GeluGradOp::InitFromOpConf() {
  CHECK(op_conf().has_gelu_grad_conf());
  EnrollInputBn("x");
  EnrollInputBn("dy");
  EnrollOutputBn("dx");
}

const PbMessage& GeluGradOp::GetCustomizedConf() const { return op_conf().gelu_grad_conf(); }

void GeluGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("dx") = *GetBlobDesc4BnInOp("x");
}

REGISTER_OP(OperatorConf::kGeluGradConf, GeluGradOp);

}  // namespace oneflow
