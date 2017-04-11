#ifndef OPERATOR_MULTINOMIAL_LOGISTIC_LOSS_OP_H_
#define OPERATOR_MULTINOMIAL_LOGISTIC_LOSS_OP_H_

#include "operator/operator.h"

namespace oneflow {

// MLLoss = MultinomialLogisticLoss

class MultinomialLogisticLossOp : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossOp);
  MultinomialLogisticLossOp() = default;
  ~MultinomialLogisticLossOp() = default;

  void Init(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return false; }
  void InferBlobDesc4ObAndDtbFromIb() const override { TODO(); }
  void InferBlobDesc4MbAndMtb() const override { TODO(); }

 private:

};

} // namespace oneflow

#endif // OPERATOR_MULTINOMIAL_LOGISTIC_LOSS_OP_H_
