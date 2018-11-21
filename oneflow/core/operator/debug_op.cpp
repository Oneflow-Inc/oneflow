#include "oneflow/core/operator/debug_op.h"

namespace oneflow {

void DebugOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void DebugOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_CPU_OP(OperatorConf::kDebugConf, DebugOp);

}  // namespace oneflow
