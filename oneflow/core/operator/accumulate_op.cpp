#include "oneflow/core/operator/accumulate_op.h"

namespace oneflow {

void AccumulateOp::InitFromOpConf() {
  CHECK(op_conf().has_accumulate_conf());

  EnrollInputBn("one", false);
  EnrollOutputBn("acc", false);
}

const PbMessage& AccumulateOp::GetCustomizedConf() const { return op_conf().accumulate_conf(); }

Maybe<void> AccumulateOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("acc") = false;
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kAccumulateConf, AccumulateOp);

}  // namespace oneflow
