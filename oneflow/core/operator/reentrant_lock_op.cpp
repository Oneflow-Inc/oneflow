#include "oneflow/core/operator/reentrant_lock_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReentrantLockOp::InitFromOpConf() {
  EnrollInputBn("start", false);
  if (op_conf().reentrant_lock_conf().has_end()) { EnrollInputBn("end", false); }
  EnrollOutputBn("out", false);
}

Maybe<void> ReentrantLockOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ_OR_RETURN(parallel_ctx->parallel_num(), 1);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->mut_shape() = Shape({1});
  const DataType data_type = GetBlobDesc4BnInOp("out")->data_type();
  CHECK_OR_RETURN(IsIntegralDataType(data_type));
  out->set_data_type(data_type);
  return Maybe<void>::Ok();
}

Maybe<void> ReentrantLockOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  return NaiveInferHasBatchDim(HasBatchDim4BnInOp);
}

void ReentrantLockOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {}

LogicalNode* ReentrantLockOp::NewProperLogicalNode() const {
  return new ReentrantLockLogicalNode();
}

REGISTER_CPU_OP(OperatorConf::kReentrantLockConf, ReentrantLockOp);

}  // namespace oneflow
