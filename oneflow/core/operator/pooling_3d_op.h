#ifndef ONEFLOW_CORE_OPERATOR_POOLING_3D_OP_H_
#define ONEFLOW_CORE_OPERATOR_POOLING_3D_OP_H_

#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

class Pooling3DOp : virtual public PoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling3DOp);
  Pooling3DOp() = default;
  virtual ~Pooling3DOp() = default;

 private:
  int32_t GetDim() const override { return 3; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_POOLING_3D_OP_H_
