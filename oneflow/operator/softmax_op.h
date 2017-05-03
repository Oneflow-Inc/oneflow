#ifndef ONEFLOW_OPERATOR_SOFTMAX_OP_H_
#define ONEFLOW_OPERATOR_SOFTMAX_OP_H_

#include "operator/operator.h"

namespace oneflow {

class SoftmaxOp : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxOp);
  SoftmaxOp() = default;
  ~SoftmaxOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  std::string GetValueFromPbOpConf(const std::string& k) const override;

  void InferShape4FwBlobs(
      std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
      ParallelPolicy policy,
      uint64_t parallel_id,
      uint64_t parallel_num) const override;
  
 private:

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_SOFTMAX_OP_H_
