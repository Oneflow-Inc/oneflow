#include "oneflow/core/operator/accumulate_op.h"

namespace oneflow {

void AccumulateOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_accumulate_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("one", false);
  EnrollOutputBn("acc", false);
}

const PbMessage& AccumulateOp::GetSpecialConf() const {
  return op_conf().accumulate_conf();
}

REGISTER_OP(OperatorConf::kAccumulateConf, AccumulateOp);

}  // namespace oneflow
