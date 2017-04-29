#include "operator/multinomial_logistic_loss_op.h"
#include "glog/logging.h"
#include "operator/operator_manager.h"

namespace oneflow {

void MultinomialLogisticLossOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_multinomial_logistic_loss_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("data");
  EnrollInputBn("label");
  EnrollOutputBn("loss", false);
  EnrollDataTmpBn("loss_buffer");
}

std::string MultinomialLogisticLossOp::GetValueFromPbOpConf(
    const std::string& k) const {
  return GetValueFromPbMessage(op_conf().multinomial_logistic_loss_conf(), k);
}

void MultinomialLogisticLossOp::InferShape4ObAndDtbFromIb() const {
}

void MultinomialLogisticLossOp::InferShape4ModelTmpBlob(ParallelPolicy policy,
    uint64_t parallel_id) const {
}

void MultinomialLogisticLossOp::InferShape4ModelDiffBlob(ParallelPolicy policy,
    uint64_t parallel_id) const {
}

REGISTER_OP(OperatorConf::kMultinomialLogisticLossConf, 
    MultinomialLogisticLossOp);

} // namespace oneflow
