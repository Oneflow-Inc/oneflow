#ifndef ONEFLOW_CORE_OPERATOR_BIAS_ADD_OP_H_
#define ONEFLOW_CORE_OPERATOR_BIAS_ADD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class BiasAddOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BiasAddOp);
  BiasAddOp() = default;
  ~BiasAddOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool IsInputBnInOpAllowedModelSplit(const std::string& ibn) const override;
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  void InferOutputBlobModelSplitAxis(
      std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BIAS_ADD_OP_H_
