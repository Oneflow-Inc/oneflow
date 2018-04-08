#ifndef ONEFLOW_CORE_OPERATOR_CONV_2D_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_2D_OP_H_

#include "oneflow/core/operator/conv_op.h"

namespace oneflow {

class Conv2DOp final : public ConvOp<2> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Conv2DOp);
  Conv2DOp() = default;
  ~Conv2DOp() = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_2D_OP_H_
