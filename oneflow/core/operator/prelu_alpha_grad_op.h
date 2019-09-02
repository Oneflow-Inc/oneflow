#ifndef ONEFLOW_CORE_OPERATOR_PRELU_ALPHA_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_PRELU_ALPHA_GRAD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class PReluAlphaGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PReluAlphaGradOp);
  PReluAlphaGradOp() = default;
  ~PReluAlphaGradOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*) const override;

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;

  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_PRELU_ALPHA_GRAD_OP_H_
