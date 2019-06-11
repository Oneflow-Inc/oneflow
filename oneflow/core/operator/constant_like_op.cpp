#include "oneflow/core/operator/constant_like_op.h"

namespace oneflow {

void ConstantLikeOp::InitFromOpConf() {
  CHECK(op_conf().has_constant_like_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& ConstantLikeOp::GetCustomizedConf() const {
  return this->op_conf().constant_like_conf();
}

void ConstantLikeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kConstantLikeConf, ConstantLikeOp);

}  // namespace oneflow
