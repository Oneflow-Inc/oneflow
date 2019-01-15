#ifndef ONEFLOW_CORE_OPERATOR_LOSS_PRINT_OP_H_
#define ONEFLOW_CORE_OPERATOR_LOSS_PRINT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LossPrintOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossPrintOp);
  LossPrintOp() = default;
  ~LossPrintOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  void InferOutBlobModelSplitAxis(std::function<int64_t*(const std::string&)> ModelSplitAxis4BnInOp,
                                  std::function<int64_t(const std::string&)> ShapeNumAxes4BnInOp,
                                  const ParallelContext* parallel_context) const override {}

  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LOSS_PRINT_OP_H_
