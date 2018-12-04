#ifndef ONEFLOW_CORE_OPERATOR_CENTER_LOSS_OP_H_
#define ONEFLOW_CORE_OPERATOR_CENTER_LOSS_OP_H_

#include "oneflow/core/operator/loss_op.h"

namespace oneflow {
  
  class CenterLossOp final : public LossOp {
   public:
    OF_DISALLOW_COPY_AND_MOVE(CenterLossOp);
    CenterLossOp() = default;
    ~CenterLossOp() = default;

    const PbMessage& GetCustomizedConf() const override;

   private:
    void VirtualInitFromOpConf() override;
    void VirtualInferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
    LossKernelConf* GetMutLossKernelConf(KernelConf*) const override;
  };
}

#endif