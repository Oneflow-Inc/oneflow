#ifndef ONEFLOW_CORE_OPERATOR_MAX_POOLING_1D_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_MAX_POOLING_1D_GRAD_OP_H_

#include "oneflow/core/operator/pooling_nd_op.h"
#include "oneflow/core/operator/max_pooling_op.h"

namespace oneflow {

class MaxPooling1DGradOp final : public PoolingNdOp<1>, public MaxPoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling1DGradOp);
  MaxPooling1DGradOp() = default;
  ~MaxPooling1DGradOp() = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MAX_POOLING_1D_GRAD_OP_H_
