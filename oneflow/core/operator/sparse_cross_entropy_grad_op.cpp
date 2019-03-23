#include "oneflow/core/operator/sparse_cross_entropy_grad_op.h"

namespace oneflow {

void SparseCrossEntropyGradOp::InitFromOpConf() {
  CHECK(op_conf().has_sparse_cross_entropy_grad_conf());
  EnrollInputBn("prediction", false);
  EnrollInputBn("label");
  EnrollInputBn("dy");
  EnrollOutputBn("prediction_diff");
}

const PbMessage& SparseCrossEntropyGradOp::GetCustomizedConf() const {
  return op_conf().sparse_cross_entropy_grad_conf();
}

void SparseCrossEntropyGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, int64_t record_piece_size,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  *GetBlobDesc4BnInOp("prediction_diff") = *GetBlobDesc4BnInOp("prediction");
}

REGISTER_OP(OperatorConf::kSparseCrossEntropyGradConf, SparseCrossEntropyGradOp);

}  // namespace oneflow
