#ifndef ONEFLOW_CORE_OPERATOR_NCCL_INTER_DEVICE_REDUCE_OP_H_
#define ONEFLOW_CORE_OPERATOR_NCCL_INTER_DEVICE_REDUCE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NcclInterDeviceReduceOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclInterDeviceReduceOp)
  NcclInterDeviceReduceOp() = default;
  ~NcclInterDeviceReduceOp() override = default;

 private:
  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext*, const OpContext*) const override;
  void InferDiffBlobDescsWithoutFwBlob(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*) const override;
  LogicalNode* NewProperLogicalNode() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NCCL_INTER_DEVICE_REDUCE_OP_H_
