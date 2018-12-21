#include "oneflow/core/operator/sqrt_op.h"

namespace oneflow {

void SqrtOp::InitFromOpConf() {
  CHECK(op_conf().has_sqrt_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& SqrtOp::GetCustomizedConf() const { return op_conf().sqrt_conf(); }

void SqrtOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kSqrtConf, SqrtOp);

}  // namespace oneflow
