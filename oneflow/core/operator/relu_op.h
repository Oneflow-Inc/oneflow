#ifndef ONEFLOW_CORE_OPERATOR_RELU_OP_H_
#define ONEFLOW_CORE_OPERATOR_RELU_OP_H_

#include <string>
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ReluOp final : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluOp);
  ReluOp() = default;
  ~ReluOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  const PbMessage& GetSpecialConf() const override;
  bool IsElemWise() const override { return true; }

  void InferShape4FwBlobs(
      std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
      ParallelPolicy policy,
      uint64_t parallel_id,
      uint64_t parallel_num) const override;

 private:
};

}  // namespace oneflow

#endif // ONEFLOW_CORE_OPERATOR_RELU_OP_H_
