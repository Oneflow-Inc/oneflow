#ifndef ONEFLOW_CORE_OPERATOR_POOLING_ND_OP_H_
#define ONEFLOW_CORE_OPERATOR_POOLING_ND_OP_H_

#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

template<int32_t NDims>
class PoolingNdOp : virtual public PoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingNdOp);
  PoolingNdOp() = default;
  virtual ~PoolingNdOp() = default;

 private:
  int32_t GetDim() const override { return NDims; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_POOLING_ND_OP_H_
