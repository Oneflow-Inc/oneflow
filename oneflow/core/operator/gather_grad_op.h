#ifndef ONEFLOW_CORE_OPERATOR_GATHER_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_GATHER_GRAD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class GatherGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherGradOp);
  GatherGradOp() = default;
  ~GatherGradOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override;
  void GetSbpSignatureRules(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_GATHER_GRAD_OP_H_
