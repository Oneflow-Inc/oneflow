#include "oneflow/core/operator/accuracy_print_op.h"

namespace oneflow {

void AccuracyPrintOp::InitFromOpConf() {
  CHECK(op_conf().has_accuracy_print_conf());
  EnrollInputBn("accuracy_acc", false);
  EnrollInputBn("batch_instance_num", false);
}

const PbMessage& AccuracyPrintOp::GetCustomizedConf() const {
  return op_conf().accuracy_print_conf();
}

REGISTER_OP(OperatorConf::kAccuracyPrintConf, AccuracyPrintOp);

}  // namespace oneflow
