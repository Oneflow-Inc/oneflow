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

void MultinomialLogisticLossOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy,
    uint64_t parallel_id,
    uint64_t parallel_num) const {
  CHECK_EQ(input_bns().size(), 2);
  CHECK_EQ(data_tmp_bns().size(), 1);
  Shape* data_shape_ptr = GetShapePtr4BnInOp(input_bns().at(0));
  Shape* label_shape_ptr = GetShapePtr4BnInOp(input_bns().at(1));

  Shape* loss_shape_ptr = GetShapePtr4BnInOp(SoleObn());
  Shape* loss_buffer_shape_ptr = GetShapePtr4BnInOp(data_tmp_bns().at(0));

  CHECK_EQ(*data_shape_ptr, *label_shape_ptr);
  *loss_shape_ptr = *data_shape_ptr;
  *loss_buffer_shape_ptr = *data_shape_ptr;
}

REGISTER_OP(OperatorConf::kMultinomialLogisticLossConf, 
    MultinomialLogisticLossOp);

} // namespace oneflow
