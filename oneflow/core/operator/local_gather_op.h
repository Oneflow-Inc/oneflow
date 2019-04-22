#ifndef ONEFLOW_CORE_OPERATOR_LOCAL_GATHER_OP_H_
#define ONEFLOW_CORE_OPERATOR_LOCAL_GATHER_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LocalGatherOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalGatherOp);
  LocalGatherOp() = default;
  ~LocalGatherOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void GetOpParallelSignatures(
      std::vector<std::unique_ptr<const OpParallelSignature>>*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LOCAL_GATHER_OP_H_
