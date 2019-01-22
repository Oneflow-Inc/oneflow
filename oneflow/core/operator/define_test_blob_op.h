#ifndef ONEFLOW_CORE_OPERATOR_DEFINE_TEST_BLOB_OP_H_
#define ONEFLOW_CORE_OPERATOR_DEFINE_TEST_BLOB_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class DefineTestBlobOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DefineTestBlobOp);
  DefineTestBlobOp() = default;
  ~DefineTestBlobOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  LogicalNode* NewProperLogicalNode() override { return new DecodeRandomLogicalNode; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void InferOutputBlobModelSplitAxis(
      std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    NaiveInferOutputBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp,
                                       parallel_context);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DEFINE_TEST_BLOB_OP_H_
