#ifndef ONEFLOW_CORE_OPERATOR_POOLING_2D_OP_H_
#define ONEFLOW_CORE_OPERATOR_POOLING_2D_OP_H_

#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

class Pooling2DOp : public PoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling2DOp);
  Pooling2DOp() = default;
  virtual ~Pooling2DOp() = default;

 protected:
  int32_t GetDim() const override { return 2; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_POOLING_2D_OP_H_
