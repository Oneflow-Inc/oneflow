#include "oneflow/core/operator/loss_print_op.h"

namespace oneflow {

void LossPrintOp::InitFromOpConf() {
  CHECK(op_conf().has_loss_print_conf());
  EnrollInputBn("loss_acc");
}

const PbMessage& LossPrintOp::GetSpecialConf() const {
  return op_conf().loss_print_conf();
}

REGISTER_OP(OperatorConf::kLossPrintConf, LossPrintOp);

}  // namespace oneflow
