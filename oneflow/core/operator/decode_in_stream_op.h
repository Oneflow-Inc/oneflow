#ifndef ONEFLOW_CORE_OPERATOR_DECODE_IN_STREAM_OP_H_
#define ONEFLOW_CORE_OPERATOR_DECODE_IN_STREAM_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class DecodeInStreamOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeInStreamOp);
  DecodeInStreamOp() = default;
  ~DecodeInStreamOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  LogicalNode* NewProperLogicalNode() override { return new DecodeInStreamLogicalNode; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DECODE_IN_STREAM_OP_H_