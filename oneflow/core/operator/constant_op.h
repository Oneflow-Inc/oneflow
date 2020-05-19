#ifndef ONEFLOW_CORE_OPERATOR_CONSTANT_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONSTANT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ConstantOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstantOp);
  ConstantOp() = default;
  ~ConstantOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx,
                             const SbpSignature* sbp_signature) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONSTANT_OP_H_
