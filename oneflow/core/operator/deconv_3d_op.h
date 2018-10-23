#ifndef ONEFLOW_CORE_OPERATOR_DECONV_3D_OP_H_
#define ONEFLOW_CORE_OPERATOR_DECONV_3D_OP_H_

#include "oneflow/core/operator/deconv_op.h"

namespace oneflow {

class Deconv3DOp final : public DeconvOp<3> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Deconv3DOp);
  Deconv3DOp() = default;
  ~Deconv3DOp() = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DECONV_3D_OP_H_
