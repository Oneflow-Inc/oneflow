#ifndef ONEFLOW_CORE_OPERATOR_CONV_3D_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_3D_OP_H_

#include "oneflow/core/operator/conv_op.h"

namespace oneflow {

class Conv3DOp final : public ConvOp<3> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Conv3DOp);
  Conv3DOp() = default;
  ~Conv3DOp() = default;

  const PbMessage& GetCustomizedConf() const override;
  ActivationType GetActivationType() const override;
  bool UseActivation() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_3D_OP_H_
