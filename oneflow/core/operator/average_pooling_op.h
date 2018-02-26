#ifndef ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_OP_H_
#define ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_OP_H_

#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

class AveragePoolingOp : virtual public PoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePoolingOp);
  AveragePoolingOp() = default;
  virtual ~AveragePoolingOp() = default;

 private:
  Pooling3DKernelConf* GetMutPooling3DKernelConf(KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_OP_H_
