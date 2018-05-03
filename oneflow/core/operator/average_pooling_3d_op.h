#ifndef ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_3D_OP_H_
#define ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_3D_OP_H_

#include "oneflow/core/operator/pooling_nd_op.h"
#include "oneflow/core/operator/average_pooling_op.h"

namespace oneflow {

class AveragePooling3DOp final : public PoolingNdOp<3>, public AveragePoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePooling3DOp);
  AveragePooling3DOp() = default;
  ~AveragePooling3DOp() = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_3D_OP_H_
