#ifndef ONEFLOW_CORE_OPERATOR_MAX_POOLING_3D_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_MAX_POOLING_3D_GRAD_OP_H_

#include "oneflow/core/operator/pooling_nd_op.h"
#include "oneflow/core/operator/max_pooling_op.h"

namespace oneflow {

class MaxPooling3DGradOp final : public PoolingNdOp<3>, public MaxPoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling3DGradOp);
  MaxPooling3DGradOp() = default;
  ~MaxPooling3DGradOp() = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MAX_POOLING_3D_GRAD_OP_H_
