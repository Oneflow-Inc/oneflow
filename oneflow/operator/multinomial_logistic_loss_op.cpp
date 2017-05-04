#include <string>
#include "operator/multinomial_logistic_loss_op.h"
#include "glog/logging.h"
#include "operator/operator_manager.h"

namespace oneflow {

void MultinomialLogisticLossOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_multinomial_logistic_loss_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("prediction");
  EnrollInputBn("label");
  EnrollOutputBn("loss", false);
  EnrollDataTmpBn("loss_buffer");
}

const PbMessage& MultinomialLogisticLossOp::GetSpecialConf() const {
  return op_conf().multinomial_logistic_loss_conf();
}

void MultinomialLogisticLossOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy,
    uint64_t parallel_id,
    uint64_t parallel_num) const {
  *GetShapePtr4BnInOp(SoleObn()) = Shape({1, 1, 1, 1});
  *GetShapePtr4BnInOp(data_tmp_bns().at(0)) = Shape({1, 1, 1, 1});
}

REGISTER_OP(OperatorConf::kMultinomialLogisticLossConf,
    MultinomialLogisticLossOp);

}  // namespace oneflow
