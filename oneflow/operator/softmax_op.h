#ifndef ONEFLOW_OPERATOR_SOFTMAX_OP_H_
#define ONEFLOW_OPERATOR_SOFTMAX_OP_H_

#include "operator/operator.h"

namespace oneflow {

class SoftmaxOp : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxOp);
  SoftmaxOp() = default;
  ~SoftmaxOp() = default;

  void Init(const OperatorConf& op_conf) override;

  void InferShape4ObAndDtbFromIb() const override { TODO(); }
  void InferShape4Mtb() const override { TODO(); }
  void InferShape4Mdb() const override { TODO(); }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_SOFTMAX_OP_H_
