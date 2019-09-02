#include "oneflow/core/operator/accumulate_op.h"

namespace oneflow {

void AccumulateOp::InitFromOpConf() {
  CHECK(op_conf().has_accumulate_conf());

  EnrollInputBn("one", false);
  EnrollOutputBn("acc", false);
}

const PbMessage& AccumulateOp::GetCustomizedConf() const { return op_conf().accumulate_conf(); }

Maybe<void> AccumulateOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  BatchAxis4BnInOp("acc")->clear_value();
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kAccumulateConf, AccumulateOp);

}  // namespace oneflow
