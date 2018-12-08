#include "oneflow/core/operator/reduce_identity_op.h"

namespace oneflow {

void ReduceIdentityOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

void ReduceIdentityOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  CHECK_EQ(GetBlobDesc4BnInOp("out")->shape().elem_cnt() % parallel_ctx->parallel_num(), 0);
}

LogicalBlobId ReduceIdentityOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(output_bn);
  return ret;
}

REGISTER_OP(OperatorConf::kReduceIdentityConf, ReduceIdentityOp);

}  // namespace oneflow
