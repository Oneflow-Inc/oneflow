#ifndef ONEFLOW_CORE_OPERATOR_CLONE_OP_H_
#define ONEFLOW_CORE_OPERATOR_CLONE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CloneOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CloneOp);
  CloneOp() = default;
  ~CloneOp() = default;

  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }
  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const;
  bool IsCloneOp() const override { return true; }

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return GetValFromCustomizedConf<std::string>("lbn");
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return op_name() + "/" + output_bn;
  }
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CLONE_OP_H_
