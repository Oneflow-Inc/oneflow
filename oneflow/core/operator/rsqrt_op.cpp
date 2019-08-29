#include "oneflow/core/operator/rsqrt_op.h"

namespace oneflow {

void RsqrtOp::InitFromOpConf() {
  CHECK(op_conf().has_rsqrt_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& RsqrtOp::GetCustomizedConf() const { return op_conf().rsqrt_conf(); }

Maybe<void> RsqrtOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kRsqrtConf, RsqrtOp);

}  // namespace oneflow
