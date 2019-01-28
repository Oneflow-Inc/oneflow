#ifndef ONEFLOW_CORE_OPERATOR_MULTIPLE_GATHER_OP_H_
#define ONEFLOW_CORE_OPERATOR_MULTIPLE_GATHER_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class MultipleGatherOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultipleGatherOp);
  MultipleGatherOp() = default;
  ~MultipleGatherOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override;
  void InferOutputBlobModelSplitAxis(
      std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MULTIPLE_GATHER_OP_H_
