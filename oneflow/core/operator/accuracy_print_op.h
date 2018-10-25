#ifndef ONEFLOW_CORE_OPERATOR_ACCURACY_PRINT_OP_H_
#define ONEFLOW_CORE_OPERATOR_ACCURACY_PRINT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class AccuracyPrintOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccuracyPrintOp);
  AccuracyPrintOp() = default;
  ~AccuracyPrintOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ACCURACY_PRINT_OP_H_
