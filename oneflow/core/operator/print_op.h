#ifndef ONEFLOW_CORE_OPERATOR_PRINT_OP_H_
#define ONEFLOW_CORE_OPERATOR_PRINT_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class PrintOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PrintOp);
  PrintOp() = default;
  ~PrintOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  LogicalNode* NewProperLogicalNode() const override { return new PrintLogicalNode; }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;

  LogicalBlobId lbi4ibn(const std::string& input_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_PRINT_OP_H_
