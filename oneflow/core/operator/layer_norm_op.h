#ifndef ONEFLOW_CORE_OPERATOR_LAYER_NORM_OP_H_
#define ONEFLOW_CORE_OPERATOR_LAYER_NORM_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LayerNormOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LayerNormOp);
  LayerNormOp() = default;
  ~LayerNormOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().layer_norm_conf(); }
  bool NeedInBlobWhenBackward() const override { return true; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext*) const override;
  void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext* parallel_ctx) const override;

 private:
  void InferOutputBlobModelSplitAxis(
      std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    NaiveInferOutputBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp,
                                       parallel_context);
  }
  void VirtualFixParallelDesc(ParallelDesc* pr_desc) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LAYER_NORM_OP_H_
