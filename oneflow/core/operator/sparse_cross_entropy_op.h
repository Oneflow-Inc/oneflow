#ifndef ONEFLOW_CORE_OPERATOR_SPARSE_CROSS_ENTROPY_OP_H_
#define ONEFLOW_CORE_OPERATOR_SPARSE_CROSS_ENTROPY_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SparseCrossEntropyOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseCrossEntropyOp);
  SparseCrossEntropyOp() = default;
  ~SparseCrossEntropyOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext*,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SPARSE_CROSS_ENTROPY_OP_H_
