#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_SCATTER_OP_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_SCATTER_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class ReduceScatterOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceScatterOp);
  ReduceScatterOp() = default;
  ~ReduceScatterOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  LogicalNode* NewProperLogicalNode() const override { return new ReduceScatterLogicalNode; }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
  Symbol<OperatorConf> GetOpConfWithoutOpNameAndLbn() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_SCATTER_OP_H_
