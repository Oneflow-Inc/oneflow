#ifndef ONEFLOW_CORE_OPERATOR_SPARSE_CROSS_ENTROPY_OP_H_
#define ONEFLOW_CORE_OPERATOR_SPARSE_CROSS_ENTROPY_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SparseCrossEntropyOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseCrossEntropyOp);
  SparseCrossEntropyOp() = default;
  ~SparseCrossEntropyOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx, int64_t record_piece_size,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;
  bool NeedOutBlobWhenBackward() const override { return false; }
  bool NeedInBlobWhenBackward() const override { return true; }

 private:
  void InferOutBlobModelSplitAxis(std::function<int64_t*(const std::string&)> ModelSplitAxis4BnInOp,
                                  std::function<int64_t(const std::string&)> ShapeNumAxes4BnInOp,
                                  const ParallelContext* parallel_context) const override {
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    NaiveInferOutBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp, parallel_context);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SPARSE_CROSS_ENTROPY_OP_H_
