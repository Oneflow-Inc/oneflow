#include <string>
#include "oneflow/core/operator/multinomial_logistic_loss_op.h"
#include "glog/logging.h"
#include "oneflow/core/operator/operator_manager.h"

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
  *GetShapePtr4BnInOp(SoleObn()) = Shape({1});
  *GetShapePtr4BnInOp(SoleDtbn()) = Shape({1});
  for (size_t i = 0;i < input_diff_bns().size(); ++i) {
    Shape* input_diff_shape_ptr = GetShapePtr4BnInOp(input_diff_bns().at(i));
    if (input_diff_shape_ptr != nullptr) {
      *input_diff_shape_ptr = *GetShapePtr4BnInOp(input_bns().at(i));
    }
  }
}

REGISTER_OP(OperatorConf::kMultinomialLogisticLossConf,
    MultinomialLogisticLossOp);

}  // namespace oneflow
