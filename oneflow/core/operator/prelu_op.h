#ifndef ONEFLOW_CORE_OPERATOR_PRELU_OP_H_
#define ONEFLOW_CORE_OPERATOR_PRELU_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class PReluOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PReluOp);
  PReluOp() = default;
  ~PReluOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return true; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext*) const override;
  void VirtualFixParallelDesc(ParallelDesc* pr_desc) const override;

 private:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_PRELU_OP_H_
