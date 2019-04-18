#ifndef ONEFLOW_CORE_OPERATOR_EVERY_NTH_OP_H_
#define ONEFLOW_CORE_OPERATOR_EVERY_NTH_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class EveryNthOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EveryNthOp);
  EveryNthOp() = default;
  ~EveryNthOp() override = default;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return true; }
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferOutputBlobTimeShape(std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
                                const ParallelContext* parallel_ctx,
                                Shape* time_shape) const override;
  void InitFromOpConf() override;
  LogicalNode* NewProperLogicalNode() const override;
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void GetSbpSignatureRules(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_EVERY_NTH_OP_H_
