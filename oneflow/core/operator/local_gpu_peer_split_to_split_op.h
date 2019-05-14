#ifndef ONEFLOW_CORE_OPERATOR_LOCAL_GPU_PEER_SPLIT_TO_SPLIT_OP_H_
#define ONEFLOW_CORE_OPERATOR_LOCAL_GPU_PEER_SPLIT_TO_SPLIT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LocalGpuPeerSplitToSplitOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalGpuPeerSplitToSplitOp);
  LocalGpuPeerSplitToSplitOp() = default;
  ~LocalGpuPeerSplitToSplitOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  LogicalNode* NewProperLogicalNode() const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LOCAL_GPU_PEER_SPLIT_TO_SPLIT_OP_H_
