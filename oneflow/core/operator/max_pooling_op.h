#ifndef ONEFLOW_CORE_OPERATOR_MAX_POOLING_OP_H_
#define ONEFLOW_CORE_OPERATOR_MAX_POOLING_OP_H_

#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

class MaxPoolingOp : virtual public PoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingOp);
  MaxPoolingOp() = default;
  virtual ~MaxPoolingOp() = default;

 private:
  PbMessage* MutableCustomizedKernelConf(KernelConf*) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MAX_POOLING_OP_H_
