#ifndef ONEFLOW_CORE_OPERATOR_RELU_OP_H_
#define ONEFLOW_CORE_OPERATOR_RELU_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class ReluOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluOp);
  ReluOp() = default;
  ~ReluOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;
  bool IsElemWiseOp() const override { return true; }

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RELU_OP_H_
