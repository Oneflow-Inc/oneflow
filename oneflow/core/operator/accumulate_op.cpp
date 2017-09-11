#include "oneflow/core/operator/accumulate_op.h"

namespace oneflow {

void AccumulateOp::InitFromOpConf() {
  CHECK(op_conf().has_accumulate_conf());

  EnrollInputBn("one", false);
  EnrollOutputBn("acc", false);
}

const PbMessage& AccumulateOp::GetSpecialConf() const {
  return op_conf().accumulate_conf();
}

REGISTER_OP(OperatorConf::kAccumulateConf, AccumulateOp);

}  // namespace oneflow
