#ifndef ONEFLOW_CORE_OPERATOR_POOLING_1D_OP_H_
#define ONEFLOW_CORE_OPERATOR_POOLING_1D_OP_H_

#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

class Pooling1DOp : virtual public PoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling1DOp);
  Pooling1DOp() = default;
  virtual ~Pooling1DOp() = default;

 private:
  int32_t GetDim() const override { return 1; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_POOLING_1D_OP_H_
