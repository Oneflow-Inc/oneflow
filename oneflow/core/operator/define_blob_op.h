#ifndef ONEFLOW_CORE_OPERATOR_DEFINE_BLOB_OP_H_
#define ONEFLOW_CORE_OPERATOR_DEFINE_BLOB_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class DefineBlobOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DefineBlobOp);
  DefineBlobOp() = default;
  ~DefineBlobOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  LogicalNode* NewProperLogicalNode() override { return new DecodeRandomLogicalNode; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DEFINE_BLOB_OP_H_
