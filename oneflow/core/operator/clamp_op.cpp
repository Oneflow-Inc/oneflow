#include "oneflow/core/operator/clamp_op.h"

namespace oneflow {

void ClampOp::InitFromOpConf() {
  CHECK(op_conf().has_clamp_conf());
  EnrollInputBn("in");
  EnrollInputBn("out");
}

const PbMessage& ClampOp::GetCustomizedConf() const { return this->op_conf().clamp_conf(); }

void ClampOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kClampConf, ClampOp);

}  // namespace oneflow
