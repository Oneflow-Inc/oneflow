#include "oneflow/core/operator/accuracy_print_op.h"

namespace oneflow {

void AccuracyPrintOp::InitFromOpConf() {
  CHECK(op_conf().has_accuracy_print_conf());
  EnrollInputBn("accuracy_acc", false);
  EnrollInputBn("accuracy_instance_num", false);
}

LogicalBlobId AccuracyPrintOp::ibn2lbi(const std::string& input_bn) const {
  if (input_bn == "accuracy_acc") {
    return op_conf().accuracy_print_conf().accuracy_lbi();
  } else if (input_bn == "accuracy_instance_num") {
    return op_conf().accuracy_print_conf().accuracy_instance_num_lbi();
  } else {
    UNIMPLEMENTED();
  }
}

const PbMessage& AccuracyPrintOp::GetCustomizedConf() const {
  return op_conf().accuracy_print_conf();
}

REGISTER_OP(OperatorConf::kAccuracyPrintConf, AccuracyPrintOp);

}  // namespace oneflow
