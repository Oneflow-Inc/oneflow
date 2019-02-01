#ifndef ONEFLOW_CORE_OPERATOR_LOG_COUNTER_OP_H_
#define ONEFLOW_CORE_OPERATOR_LOG_COUNTER_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class LogCounterOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogCounterOp);
  LogCounterOp() = default;
  ~LogCounterOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  virtual LogicalNode* NewProperLogicalNode() { return new PrintLogicalNode; }

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return true; }
  void InferOutputBlobLbpdHint(std::function<LbpdHint*(const std::string&)> LbpdHint4BnInOp,
                               std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
                               const ParallelContext* parallel_context) const override {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LOG_COUNTER_OP_H_
