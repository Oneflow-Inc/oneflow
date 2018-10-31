#include "oneflow/core/operator/loss_print_op.h"

namespace oneflow {

void LossPrintOp::InitFromOpConf() {
  CHECK(op_conf().has_loss_print_conf());
  EnrollInputBn("loss_acc", false);
  EnrollInputBn("loss_instance_num", false);
  if (op_conf().loss_print_conf().has_reduction_lbi()) { EnrollInputBn("reduction_acc"); }
}

LogicalBlobId LossPrintOp::ibn2lbi(const std::string& input_bn) const {
  if (input_bn == "loss_acc") {
    return op_conf().loss_print_conf().loss_lbi();
  } else if (input_bn == "loss_instance_num") {
    return op_conf().loss_print_conf().loss_instance_num_lbi();
  } else if (input_bn == "reduction_acc") {
    return op_conf().loss_print_conf().reduction_lbi();
  } else {
    UNIMPLEMENTED();
  }
}

const PbMessage& LossPrintOp::GetCustomizedConf() const { return op_conf().loss_print_conf(); }

REGISTER_CPU_OP(OperatorConf::kLossPrintConf, LossPrintOp);

}  // namespace oneflow
