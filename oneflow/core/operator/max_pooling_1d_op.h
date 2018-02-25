#ifndef ONEFLOW_CORE_OPERATOR_MAX_POOLING_1D_OP_H_
#define ONEFLOW_CORE_OPERATOR_MAX_POOLING_1D_OP_H_

#include "oneflow/core/operator/pooling_1d_op.h"
#include "oneflow/core/operator/max_pooling_op.h"

namespace oneflow {

class MaxPooling1DOp final : public Pooling1DOp, public MaxPoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling1DOp);
  MaxPooling1DOp() = default;
  ~MaxPooling1DOp() = default;

  const PbMessage& GetSpecialConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MAX_POOLING_1D_OP_H_
