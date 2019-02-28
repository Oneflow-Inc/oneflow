#ifndef ONEFLOW_CORE_OPERATOR_BINARY_CROSS_ENTROPY_OP_H_
#define ONEFLOW_CORE_OPERATOR_BINARY_CROSS_ENTROPY_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class BinaryCrossEntropyOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BinaryCrossEntropyOp);
  BinaryCrossEntropyOp() = default;
  ~BinaryCrossEntropyOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx, int64_t record_piece_size,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;
  bool NeedOutBlobWhenBackward() const override { return false; }
  bool NeedInBlobWhenBackward() const override { return true; }

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BINARY_CROSS_ENTROPY_OP_H_
