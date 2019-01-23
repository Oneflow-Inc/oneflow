#ifndef ONEFLOW_CORE_OPERATOR_HINGE_LOSS_OP_H_
#define ONEFLOW_CORE_OPERATOR_HINGE_LOSS_OP_H_

#include "oneflow/core/operator/loss_op.h"

namespace oneflow {

class HingeLossOp final : public LossOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HingeLossOp);
  HingeLossOp() = default;
  ~HingeLossOp() = default;

  const PbMessage& GetCustomizedConf() const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void InferOutputBlobModelSplitAxis(
      std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    NaiveInferOutputBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp,
                                       parallel_context);
  }

  void VirtualInitFromOpConf() override;
  void VirtualInferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  LossKernelConf* GetMutLossKernelConf(KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_HINGE_LOSS_H_
