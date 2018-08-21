#ifndef ONEFLOW_CORE_OPERATOR_SMOOTH_L1_LOSS_OP_H_
#define ONEFLOW_CORE_OPERATOR_SMOOTH_L1_LOSS_OP_H_

#include "oneflow/core/operator/loss_op.h"

namespace oneflow {

class SmoothL1LossOp final : public LossOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SmoothL1LossOp);
  SmoothL1LossOp() = default;
  ~SmoothL1LossOp() = default;

  const PbMessage& GetCustomizedConf() const override;

 private:
  void VirtualInitFromOpConf() override;
  void VirtualInferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  LossKernelConf* GetMutLossKernelConf(KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SMOOTH_L1_LOSS_OP_H_
