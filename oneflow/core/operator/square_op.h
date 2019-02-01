#ifndef ONEFLOW_CORE_OPERATOR_SQUARE_H_
#define ONEFLOW_CORE_OPERATOR_SQUARE_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SquareOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SquareOp);
  SquareOp() = default;
  ~SquareOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedOutBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return true; }
  void InferOutputBlobLbpdHint(std::function<LbpdHint*(const std::string&)> LbpdHint4BnInOp,
                               std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
                               const ParallelContext* parallel_context) const override {
    NaiveInferOutputBlobLbpdHint(LbpdHint4BnInOp, ShapeNumAxes4BnInOp, parallel_context);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SQUARE_H_
