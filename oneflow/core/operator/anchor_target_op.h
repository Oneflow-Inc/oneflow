#ifndef ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_OP_H_
#define ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class AnchorTargetOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AnchorTargetOp);
  AnchorTargetOp() = default;
  ~AnchorTargetOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
  const DataType GetDataTypeFromInputPb(const BlobDesc* gt_boxes_blob_desc) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_OP_H_
