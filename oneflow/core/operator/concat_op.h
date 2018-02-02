#ifndef ONEFLOW_CORE_OPERATOR_CONCAT_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONCAT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ConcatOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConcatOp);
  ConcatOp() = default;
  ~ConcatOp() = default;

  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }
  void InitFromOpConf() override;

  const PbMessage& GetSpecialConf() const override;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return ibn2lbn_.at(input_bn);
  }
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;

  HashMap<std::string, std::string> ibn2lbn_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONCAT_OP_H_
