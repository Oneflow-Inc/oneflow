#include "oneflow/core/operator/square_op.h"

namespace oneflow {

void SquareOp::InitFromOpConf() {
  CHECK(op_conf().has_square_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& SquareOp::GetCustomizedConf() const { return op_conf().square_conf(); }

void SquareOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kSquareConf, SquareOp);

}  // namespace oneflow
