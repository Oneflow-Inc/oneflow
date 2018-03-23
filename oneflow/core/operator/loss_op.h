#ifndef ONEFLOW_CORE_OPERATOR_LOSS_OP_H_
#define ONEFLOW_CORE_OPERATOR_LOSS_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LossOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossOp);
  LossOp() = default;
  virtual ~LossOp() = default;

  void InitFromOpConf() override;
  bool IsLossOp() const override { return true; }

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 protected:
  virtual void VirtualInitFromOpConf() {}
  virtual void VirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {}
  virtual LossKernelConf* GetMutLossKernelConf(KernelConf*) const = 0;

  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;

 private:
  std::string obn2lbn(const std::string& output_bn) const override {
    if (output_bn == "reduction_coefficient") {
      return op_name() + "/reduction_coefficient";
    } else {
      return op_name() + "/" + GetValFromCustomizedConf<std::string>(output_bn);
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SOFTMAX_LOSS_OP_H_
