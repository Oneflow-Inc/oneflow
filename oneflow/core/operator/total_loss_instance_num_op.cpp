#include "oneflow/core/operator/total_loss_instance_num_op.h"

namespace oneflow {

void TotalLossInstanceNumOp::VirtualInitFromOpConf() {
  CHECK(op_conf().has_total_loss_instance_num_conf());
}

Maybe<void> TotalLossInstanceNumOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  for (const std::string& ibn : input_bns()) {
    CHECK_OR_RETURN(*GetBlobDesc4BnInOp(ibn) == *GetBlobDesc4BnInOp(input_bns().Get(0)));
  }
  return Maybe<void>::Ok();
}

const PbMessage& TotalLossInstanceNumOp::GetCustomizedConf() const {
  return op_conf().total_loss_instance_num_conf();
}

Maybe<void> TotalLossInstanceNumOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const auto& ibn : input_bns()) {
    CHECK_EQ_OR_RETURN(BatchAxis4BnInOp(ibn)->has_value(), false);
  }
  BatchAxis4BnInOp("out")->clear_value();
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kTotalLossInstanceNumConf, TotalLossInstanceNumOp);

}  // namespace oneflow
