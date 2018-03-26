#include "oneflow/core/operator/sigmoid_op.h"

namespace oneflow {

void SigmoidOp::InitFromOpConf() {
  CHECK(op_conf().has_sigmoid_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& SigmoidOp::GetCustomizedConf() const {
  return op_conf().sigmoid_conf();
}

void SigmoidOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kSigmoidConf, SigmoidOp);

}  // namespace oneflow
