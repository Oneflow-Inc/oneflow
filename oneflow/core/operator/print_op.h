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
  virtual LogicalNode* NewProperLogicalNode() { return new PrintLogicalNode; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override {}

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return true; }
  void InferOutputBlobSbpInferHint(
      std::function<SbpInferHint*(const std::string&)> SbpInferHint4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {}

  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_PRINT_OP_H_
