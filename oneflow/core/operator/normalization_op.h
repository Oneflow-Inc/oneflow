#ifndef ONEFLOW_CORE_OPERATOR_NORMALIZATION_OP_H_
#define ONEFLOW_CORE_OPERATOR_NORMALIZATION_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NormalizationOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationOp);
  NormalizationOp() = default;
  ~NormalizationOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedOutBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void GetSbpSignatureRules(std::vector<std::unique_ptr<const SbpSignatureRule>>*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NORMALIZATION_OP_H_
