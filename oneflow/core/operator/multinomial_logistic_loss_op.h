#ifndef ONEFLOW_CORE_OPERATOR_MULTINOMIAL_LOGISTIC_LOSS_OP_H_
#define ONEFLOW_CORE_OPERATOR_MULTINOMIAL_LOGISTIC_LOSS_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

// MLLoss = MultinomialLogisticLoss

class MultinomialLogisticLossOp : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossOp);
  MultinomialLogisticLossOp() = default;
  ~MultinomialLogisticLossOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  const PbMessage& GetSpecialConf() const override;
  bool IsLossOp() const override { return true; }
  
  void InferShape4FwBlobs(
      std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
      ParallelPolicy policy,
      int64_t parallel_id,
      int64_t parallel_num) const override;

 private:

};

} // namespace oneflow

#endif // ONEFLOW_CORE_OPERATOR_MULTINOMIAL_LOGISTIC_LOSS_OP_H_
