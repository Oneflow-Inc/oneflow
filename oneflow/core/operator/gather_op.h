#ifndef ONEFLOW_CORE_OPERATOR_GATHER_OP_H_
#define ONEFLOW_CORE_OPERATOR_GATHER_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class GatherOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherOp);
  GatherOp() = default;
  ~GatherOp() = default;

 private:
  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;
  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }
  bool IsGatherOp() const override { return true; }
  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;
  void VirtualFixParallelDesc(ParallelDesc* pr_desc) const override;
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_GATHER_OP_H_
