#ifndef ONEFLOW_CORE_OPERATOR_LAYER_NORMALIZATION_OP_H_
#define ONEFLOW_CORE_OPERATOR_LAYER_NORMALIZATION_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LayerNormOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LayerNormOp);
  LayerNormOp() = default;
  ~LayerNormOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().layer_norm_conf(); }
  // TODO: not sure NeedOut/InBlobWhenBw() ?

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LAYER_NORMALIZATION_OP_H_
