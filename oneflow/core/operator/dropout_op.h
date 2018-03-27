#ifndef ONEFLOW_CORE_OPERATOR_DROPOUT_OP_H_
#define ONEFLOW_CORE_OPERATOR_DROPOUT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class DropoutOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DropoutOp);
  DropoutOp() = default;
  ~DropoutOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool IsElemWiseOp() const override { return true; }
  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }
  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DROPOUT_OP_H_
