#ifndef ONEFLOW_CORE_OPERATOR_REPEAT_OP_H_
#define ONEFLOW_CORE_OPERATOR_REPEAT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class RepeatOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RepeatOp);
  RepeatOp() = default;
  ~RepeatOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InitFromOpConf() override;
  LogicalNode* NewProperLogicalNode() override;
  bool NeedInBlobWhenBackward() const override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REPEAT_OP_H_
