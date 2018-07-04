#ifndef ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_1D_OP_H_
#define ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_1D_OP_H_

#include "oneflow/core/operator/pooling_nd_op.h"
#include "oneflow/core/operator/average_pooling_op.h"

namespace oneflow {

class AveragePooling1DOp final : public PoolingNdOp<1>, public AveragePoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePooling1DOp);
  AveragePooling1DOp() = default;
  ~AveragePooling1DOp() = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_1D_OP_H_
