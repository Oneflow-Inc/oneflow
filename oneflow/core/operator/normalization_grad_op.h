#ifndef ONEFLOW_CORE_OPERATOR_NORMALIZATION_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_NORMALIZATION_GRAD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NormalizationGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationGradOp);
  NormalizationGradOp() = default;
  ~NormalizationGradOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void GetSbpSignatures(std::vector<std::unique_ptr<const SbpSignature>>*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NORMALIZATION_GRAD_OP_H_
