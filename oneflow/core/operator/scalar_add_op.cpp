#include "oneflow/core/operator/scalar_add_op.h"

namespace oneflow {

void ScalarAddOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void ScalarAddOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kScalarAddConf, ScalarAddOp);

}  // namespace oneflow
