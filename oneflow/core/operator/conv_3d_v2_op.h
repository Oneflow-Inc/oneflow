#ifndef ONEFLOW_CORE_OPERATOR_CONV_3D_V2_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_3D_V2_OP_H_

#include "oneflow/core/operator/conv_op.h"

namespace oneflow {

class Conv3DV2Op final : public ConvOp<3> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Conv3DV2Op);
  Conv3DV2Op() = default;
  ~Conv3DV2Op() = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_3D_V2_OP_H_
