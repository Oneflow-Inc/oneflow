#include "oneflow/core/operator/identity_op.h"

namespace oneflow {

void IdentityOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void IdentityOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

}  // namespace oneflow
