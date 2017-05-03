#ifndef ONEFLOW_OPERATOR_RELU_OP_H_
#define ONEFLOW_OPERATOR_RELU_OP_H_

#include <string>
#include "operator/operator.h"

namespace oneflow {

class ReluOp final : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluOp);
  ReluOp() = default;
  ~ReluOp() = default;

  std::string GetValueFromPbOpConf(const std::string& k) const override;
  void InitFromOpConf(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return true; }

  void InferShape4FwBlobs(
      std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
      ParallelPolicy policy,
      uint64_t parallel_id,
      uint64_t parallel_size) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_OPERATOR_RELU_OP_H_
