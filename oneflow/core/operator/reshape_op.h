#ifndef ONEFLOW_CORE_OPERATOR_RESHAPE_OP_H_
#define ONEFLOW_CORE_OPERATOR_RESHAPE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ReshapeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReshapeOp);
  ReshapeOp() = default;
  ~ReshapeOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool IsElemWiseOp() const override { return false; }
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  bool IsForwardInplace() const override { return true; }
  bool IsBackwardInplace() const override { return true; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  void InferOutBlobModelSplitAxis(std::function<int64_t*(const std::string&)> ModelSplitAxis4BnInOp,
                                  std::function<int64_t(const std::string&)> ShapeNumAxes4BnInOp,
                                  const ParallelContext* parallel_context) const override {
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    NaiveInferOutBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp, parallel_context);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RESHAPE_OP_H_
