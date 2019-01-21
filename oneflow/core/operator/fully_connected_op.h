#ifndef ONEFLOW_CORE_OPERATOR_FULLY_CONNECTED_OP_H_
#define ONEFLOW_CORE_OPERATOR_FULLY_CONNECTED_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class FullyConnectedOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FullyConnectedOp);
  FullyConnectedOp() = default;
  ~FullyConnectedOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  int32_t ModelSplitAxis() const override { return 1; }
  int32_t MaxModelSplitNum() const override { return op_conf().fully_connected_conf().units(); }

 private:
  void InferOutputBlobModelSplitAxis(
      std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    NaiveInferOutputBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp,
                                       parallel_context);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_FULLY_CONNECTED_OP_H_
