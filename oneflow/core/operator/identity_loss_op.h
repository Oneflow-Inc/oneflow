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
  void InferOutBlobModelSplitAxis(std::function<int64_t*(const std::string&)> ModelSplitAxis4BnInOp,
                                  std::function<int64_t(const std::string&)> ShapeNumAxes4BnInOp,
                                  const ParallelContext* parallel_context) const override {
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    NaiveInferOutBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp, parallel_context);
  }

  LossKernelConf* GetMutLossKernelConf(KernelConf*) const override;
  void VirtualInitFromOpConf() override;
  void VirtualInferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_IDENTITY_LOSS_OP_H_
