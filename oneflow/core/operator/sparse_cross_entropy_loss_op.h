#ifndef ONEFLOW_CORE_OPERATOR_SPARSE_CROSS_ENTROPY_LOSS_OP_H_
#define ONEFLOW_CORE_OPERATOR_SPARSE_CROSS_ENTROPY_LOSS_OP_H_

#include "oneflow/core/operator/loss_op.h"

namespace oneflow {

class SparseCrossEntropyLossOp final : public LossOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseCrossEntropyLossOp);
  SparseCrossEntropyLossOp() = default;
  ~SparseCrossEntropyLossOp() = default;

  const PbMessage& GetCustomizedConf() const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void InferOutputBlobSbpInferHint(
      std::function<SbpInferHint*(const std::string&)> SbpInferHint4BnInOp,
      const ParallelContext* parallel_context) const override {
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    NaiveInferOutputBlobSbpInferHint(SbpInferHint4BnInOp, parallel_context);
  }

  LossKernelConf* GetMutLossKernelConf(KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SPARSE_CROSS_ENTROPY_LOSS_OP_H_
