#include "oneflow/core/operator/non_maximum_supression_op.h"

namespace oneflow {

void NonMaximumSupressionOp::InitFromOpConf() {
  CHECK(op_conf().has_non_maximum_supression_conf());
  EnrollInputBn("boxes", false);
  EnrollInputBn("scores", false);
  EnrollOutputBn("out", false);
}

const PbMessage& NonMaximumSupressionOp::GetCustomizedConf() const {
  return this->op_conf().non_maximum_supression_conf();
}

void NonMaximumSupressionOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // TODO
}

REGISTER_OP(OperatorConf::kNonMaximumSupressionConf, NonMaximumSupressionOp);

}  // namespace oneflow
