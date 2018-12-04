#include "oneflow/core/operator/reduce_inplace_identity_op.h"

namespace oneflow {

void ReduceInplaceIdentityOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void ReduceInplaceIdentityOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

LogicalBlobId ReduceInplaceIdentityOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(output_bn);
  return ret;
}

REGISTER_OP(OperatorConf::kReduceInplaceIdentityConf, ReduceInplaceIdentityOp);

}  // namespace oneflow
