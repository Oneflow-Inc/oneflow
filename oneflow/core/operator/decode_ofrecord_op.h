#ifndef ONEFLOW_CORE_OPERATOR_DECODE_OFRECORD_OP_H_
#define ONEFLOW_CORE_OPERATOR_DECODE_OFRECORD_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class DecodeOFRecordOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeOFRecordOp);
  DecodeOFRecordOp() = default;
  ~DecodeOFRecordOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  LogicalNode* NewProperLogicalNode() override { return new DecodeLogicalNode; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }

  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DECODE_OFRECORD_OP_H_
