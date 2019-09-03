#include "oneflow/core/operator/wait_and_send_ids_op.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void WaitAndSendIdsOp::InitFromOpConf() {
  CHECK(op_conf().has_wait_and_send_ids_conf());
  EnrollOutputBn("out", false);
}

LogicalNode* WaitAndSendIdsOp::NewProperLogicalNode() const {
  return new WaitAndSendIdsLogicalNode();
}

const PbMessage& WaitAndSendIdsOp::GetCustomizedConf() const {
  return op_conf().wait_and_send_ids_conf();
}

Maybe<void> WaitAndSendIdsOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ_OR_RETURN(parallel_ctx->parallel_num(), 1);
  GetBlobDesc4BnInOp("out")->mut_shape() = Shape({1});
  GetBlobDesc4BnInOp("out")->set_data_type(op_conf().wait_and_send_ids_conf().data_type());
  return Maybe<void>::Ok();
}

Maybe<void> WaitAndSendIdsOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  BatchAxis4BnInOp("out")->clear_value();
  return Maybe<void>::Ok();
}

void WaitAndSendIdsOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().Split(output_bns(), 0).Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_CPU_OP(OperatorConf::kWaitAndSendIdsConf, WaitAndSendIdsOp);

}  // namespace oneflow
