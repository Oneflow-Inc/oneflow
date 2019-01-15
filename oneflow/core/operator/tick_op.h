#ifndef ONEFLOW_CORE_OPERATOR_TICK_OP_H_
#define ONEFLOW_CORE_OPERATOR_TICK_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class TickOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TickOp);
  TickOp() = default;
  ~TickOp() = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().tick_conf(); }

 private:
  void InferOutBlobModelSplitAxis(std::function<int64_t*(const std::string&)> ModelSplitAxis4BnInOp,
                                  std::function<int64_t(const std::string&)> ShapeNumAxes4BnInOp,
                                  const ParallelContext* parallel_context) const override {
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    NaiveInferOutBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp, parallel_context);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_TICK_OP_H_
