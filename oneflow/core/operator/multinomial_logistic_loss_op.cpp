#include "oneflow/core/operator/multinomial_logistic_loss_op.h"

namespace oneflow {

void MultinomialLogisticLossOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_multinomial_logistic_loss_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("prediction");
  EnrollInputBn("label", false);
  EnrollOutputBn("loss", false);
  EnrollDataTmpBn("loss_buffer");
}

const PbMessage& MultinomialLogisticLossOp::GetSpecialConf() const {
  return op_conf().multinomial_logistic_loss_conf();
}

void MultinomialLogisticLossOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  *GetShapePtr4BnInOp("loss") = Shape({1});
  *GetShapePtr4BnInOp("loss_buffer") = Shape({1});
}

REGISTER_OP(OperatorConf::kMultinomialLogisticLossConf,
            MultinomialLogisticLossOp);

}  // namespace oneflow
