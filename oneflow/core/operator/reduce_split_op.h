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
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    return Maybe<void>::Ok();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_SPLIT_OP_H_
