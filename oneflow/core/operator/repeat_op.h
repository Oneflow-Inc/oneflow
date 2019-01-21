#ifndef ONEFLOW_CORE_OPERATOR_REPEAT_OP_H_
#define ONEFLOW_CORE_OPERATOR_REPEAT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class RepeatOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RepeatOp);
  RepeatOp() = default;
  ~RepeatOp() override = default;

  int32_t GetRepeatNum() const;

 private:
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferOutputBlobTimeShape(std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
                                const ParallelContext* parallel_ctx,
                                Shape* time_shape) const override;

  void InferDiffBlobDescsWithoutFwBlob(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*) const override;
  void InitFromOpConf() override;
  LogicalNode* NewProperLogicalNode() override;
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
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

#endif  // ONEFLOW_CORE_OPERATOR_REPEAT_OP_H_
