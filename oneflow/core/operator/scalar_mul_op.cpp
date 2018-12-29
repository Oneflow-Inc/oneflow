#include "oneflow/core/operator/scalar_mul_op.h"

namespace oneflow {

void ScalarMulOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void ScalarMulOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kScalarMulConf, ScalarMulOp);

}  // namespace oneflow
