#include "oneflow/core/operator/identity_op.h"

namespace oneflow {

void IdentityOp::InitFromOpConf() {
  CHECK(op_conf().has_identity_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& IdentityOp::GetCustomizedConf() const { return op_conf().identity_conf(); }

void IdentityOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kIdentityConf, IdentityOp);

}  // namespace oneflow
