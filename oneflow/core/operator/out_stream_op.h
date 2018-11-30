#ifndef ONEFLOW_CORE_OPERATOR_OUT_STREAM_OP_H_
#define ONEFLOW_CORE_OPERATOR_OUT_STREAM_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class OutStreamOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OutStreamOp);
  OutStreamOp() = default;
  ~OutStreamOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  virtual LogicalNode* NewProperLogicalNode() { return new OutStreamLogicalNode; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override {}

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_OUT_STREAM_OP_H_