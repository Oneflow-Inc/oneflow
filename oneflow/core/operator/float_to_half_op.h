#ifndef ONEFLOW_CORE_OPERATOR_FLOAT_TO_HALF_OP_H_
#define ONEFLOW_CORE_OPERATOR_FLOAT_TO_HALF_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class FloatToHalfOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FloatToHalfOp);
  FloatToHalfOp() = default;
  ~FloatToHalfOp() = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().float_to_half_conf(); }
  bool IsElemWiseOp() const override { return true; }
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return true; }
  void InferOutputBlobModelSplitAxis(
      std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    NaiveInferOutputBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp,
                                       parallel_context);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_FLOAT_TO_HALF_OP_H_
