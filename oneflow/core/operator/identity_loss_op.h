#ifndef ONEFLOW_CORE_OPERATOR_IDENTITY_LOSS_OP_H_
#define ONEFLOW_CORE_OPERATOR_IDENTITY_LOSS_OP_H_

#include "oneflow/core/operator/loss_op.h"

namespace oneflow {

class IdentityLossOp final : public LossOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdentityLossOp);
  IdentityLossOp() = default;
  ~IdentityLossOp() override = default;

  const PbMessage& GetCustomizedConf() const override;

 private:
  void GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;

  LossKernelConf* GetMutLossKernelConf(KernelConf*) const override;
  Maybe<void> VirtualInferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_IDENTITY_LOSS_OP_H_
