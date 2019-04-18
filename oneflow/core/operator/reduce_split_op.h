#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_SPLIT_OP_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_SPLIT_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class ReduceSplitOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceSplitOp);
  ReduceSplitOp() = default;
  ~ReduceSplitOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  LogicalNode* NewProperLogicalNode() const override { return new ReduceSplitLogicalNode; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { UNIMPLEMENTED(); }
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*) const override;
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_SPLIT_OP_H_
