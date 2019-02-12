#ifndef ONEFLOW_CORE_OPERATOR_SIGMOID_CROSS_ENTROPY_LOSS_OP_H_
#define ONEFLOW_CORE_OPERATOR_SIGMOID_CROSS_ENTROPY_LOSS_OP_H_

#include "oneflow/core/operator/loss_op.h"

namespace oneflow {

class SigmoidCrossEntropyLossOp final : public LossOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SigmoidCrossEntropyLossOp);
  SigmoidCrossEntropyLossOp() = default;
  ~SigmoidCrossEntropyLossOp() = default;

  const PbMessage& GetCustomizedConf() const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void InferOutputBlobSbpInferHint(
      std::function<SbpInferHint*(const std::string&)> SbpInferHint4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    NaiveInferOutputBlobSbpInferHint(SbpInferHint4BnInOp, ShapeNumAxes4BnInOp, parallel_context);
  }

  void VirtualInitFromOpConf() override;
  void VirtualInferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  LossKernelConf* GetMutLossKernelConf(KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SIGMOID_CROSS_ENTROPY_LOSS_OP_H_
