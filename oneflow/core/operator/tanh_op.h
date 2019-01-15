#ifndef ONEFLOW_CORE_OPERATOR_TANH_OP_H_
#define ONEFLOW_CORE_OPERATOR_TANH_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class TanHOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TanHOp);
  TanHOp() = default;
  ~TanHOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool IsElemWiseOp() const override { return true; }
  bool NeedInBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  void InferOutBlobModelSplitAxis(std::function<int64_t*(const std::string&)> ModelSplitAxis4BnInOp,
                                  std::function<int64_t(const std::string&)> ShapeNumAxes4BnInOp,
                                  const ParallelContext* parallel_context) const override {
    NaiveInferOutBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp, parallel_context);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_TANH_OP_H_
