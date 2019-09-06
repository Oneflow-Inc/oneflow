#ifndef ONEFLOW_CORE_OPERATOR_FOREIGN_INPUT_OP_H_
#define ONEFLOW_CORE_OPERATOR_FOREIGN_INPUT_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class ForeignInputOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForeignInputOp);
  ForeignInputOp() : Operator() {}
  ~ForeignInputOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  LogicalNode* NewProperLogicalNode() const override { return new ForeignInputLogicalNode; }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_FOREIGN_INPUT_OP_H_
