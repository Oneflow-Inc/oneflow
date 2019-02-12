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
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void InferOutputBlobSbpInferHint(
      std::function<SbpInferHint*(const std::string&)> SbpInferHint4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {}

  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ACCURACY_PRINT_OP_H_
