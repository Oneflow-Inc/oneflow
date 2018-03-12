#include "oneflow/core/operator/gather_op.h"

namespace oneflow {

void GatherOp::InitFromOpConf() {
  CHECK(op_conf().has_gather_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& GatherOp::GetCustomizedConf() const {
  return op_conf().gather_conf();
}

void GatherOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kGatherConf, GatherOp);

}  // namespace oneflow
