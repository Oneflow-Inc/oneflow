#ifndef ONEFLOW_CORE_OPERATOR_MEAN_OP_H_
#define ONEFLOW_CORE_OPERATOR_MEAN_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class MeanOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MeanOp);
  MeanOp() = default;
  ~MeanOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().mean_conf(); }
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext*) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void InferOutputBlobLbpdHint(std::function<LbpdHint*(const std::string&)> LbpdHint4BnInOp,
                               std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
                               const ParallelContext* parallel_context) const override {
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    NaiveInferOutputBlobLbpdHint(LbpdHint4BnInOp, ShapeNumAxes4BnInOp, parallel_context);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MEAN_OP_H_
