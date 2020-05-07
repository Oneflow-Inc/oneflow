#ifndef ONEFLOW_CORE_OPERATOR_SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_OP_H_
#define ONEFLOW_CORE_OPERATOR_SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_OP_H_

#include "oneflow/core/operator/loss_op.h"

namespace oneflow {

class SparseSoftmaxCrossEntropyLossOp final : public LossOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseSoftmaxCrossEntropyLossOp);
  SparseSoftmaxCrossEntropyLossOp() = default;
  ~SparseSoftmaxCrossEntropyLossOp() = default;

  const PbMessage& GetCustomizedConf() const override;

 private:
  void VirtualInitFromOpConf() override;
  Maybe<void> VirtualInferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const override;
  LossKernelConf* GetMutLossKernelConf(KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_OP_H_
