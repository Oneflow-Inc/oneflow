#include "oneflow/core/operator/tanh_op.h"

namespace oneflow {

void TanHOp::InitFromOpConf() {
  CHECK(op_conf().has_tanh_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& TanHOp::GetCustomizedConf() const { return op_conf().tanh_conf(); }

void TanHOp::InferBlobDescs(std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kTanhConf, TanHOp);

}  // namespace oneflow
