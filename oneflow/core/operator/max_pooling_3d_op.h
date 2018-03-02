#ifndef ONEFLOW_CORE_OPERATOR_MAX_POOLING_3D_OP_H_
#define ONEFLOW_CORE_OPERATOR_MAX_POOLING_3D_OP_H_

#include "oneflow/core/operator/pooling_3d_op.h"
#include "oneflow/core/operator/max_pooling_op.h"

namespace oneflow {

class MaxPooling3DOp final : public Pooling3DOp, public MaxPoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling3DOp);
  MaxPooling3DOp() = default;
  ~MaxPooling3DOp() = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MAX_POOLING_3D_OP_H_
