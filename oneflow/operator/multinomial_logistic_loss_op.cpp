#include "operator/multinomial_logistic_loss_op.h"
#include "glog/logging.h"

namespace oneflow {

void MultinomialLogisticLossOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_multinomial_logistic_loss_conf());
  mut_op_conf() = op_conf;
}
std::string MultinomialLogisticLossOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().multinomial_logistic_loss_conf(), k);
}
} // namespace oneflow
