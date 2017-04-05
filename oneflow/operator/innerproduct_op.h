#ifndef OPERATOR_INNERPRODUCT_OP_H_
#define OPERATOR_INNERPRODUCT_OP_H_

#include "operator/operator.h"

namespace oneflow {

class InnerProductOp final : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InnerProductOp);
  InnerProductOp() = default;
  ~InnerProductOp() = default;

  void Init(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return false; }

 private:

};

} // namespace oneflow

#endif // OPERATOR_INNERPRODUCT_OP_H_
