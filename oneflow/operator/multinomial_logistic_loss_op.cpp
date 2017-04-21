#include "operator/multinomial_logistic_loss_op.h"
#include "glog/logging.h"

namespace oneflow {

void MultinomialLogisticLossOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();
  
  CHECK(op_conf.has_multinomial_logistic_loss_conf());
  auto cnf = new MultinomialLogisticLossOpConf(
    op_conf.multinomial_logistic_loss_conf());
  mut_pb_op_conf().reset(cnf);

  EnrollInputBn("data");
  EnrollInputBn("label");
  EnrollOutputBn("loss", false);
  EnrollDataTmpBn("loss_buffer");
}

} // namespace oneflow
