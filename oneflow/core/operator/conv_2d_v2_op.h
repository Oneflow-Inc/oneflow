#ifndef ONEFLOW_CORE_OPERATOR_CONV_2D_V2_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_2D_V2_OP_H_

#include "oneflow/core/operator/conv_v2_op.h"

namespace oneflow {

class Conv2DV2Op final : public ConvV2Op<2> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Conv2DV2Op);
  Conv2DV2Op() = default;
  ~Conv2DV2Op() = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_2D_V2_OP_H_
