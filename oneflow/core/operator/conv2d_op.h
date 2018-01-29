#ifndef ONEFLOW_CORE_OPERATOR_CONV2D_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV2D_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class Conv2dOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Conv2dOp);
  Conv2dOp() = default;
  ~Conv2dOp() = default;

  void InitFromOpConf() override;

  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }
  const PbMessage& GetSpecialConf() const override;
  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

  int32_t ModelSplitAxis() const override { return 1; }
  int32_t MaxModelSplitNum() const override {
    return op_conf().conv2d_conf().out_num();
  }
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV2D_OP_H_
