#ifndef ONEFLOW_CORE_OPERATOR_STOCHASTIC_POOLING_OP_H_
#define ONEFLOW_CORE_OPERATOR_STOCHASTIC_POOLING_OP_H_

#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

class StochasticPoolingOp final : public PoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StochasticPoolingOp);
  StochasticPoolingOp() = default;
  ~StochasticPoolingOp() = default;

  const PbMessage& GetSpecialConf() const override;

 private:
  void VirtualEnrollDataTmpBn() override;
  void VirtualInferDataTmpBlobDesc(std::function<BlobDesc*(const std::string)>
                                       GetBlobDesc4BnInOp) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_STOCHASTIC_POOLING_OP_H_
