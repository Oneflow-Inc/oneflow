#ifndef ONEFLOW_CORE_OPERATOR_DECONV_2D_OP_H_
#define ONEFLOW_CORE_OPERATOR_DECONV_2D_OP_H_

#include "oneflow/core/operator/deconv_op.h"

namespace oneflow {

class Deconv2DOp final : public DeconvOp<2> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Deconv2DOp);
  Deconv2DOp() = default;
  ~Deconv2DOp() = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DECONV_2D_OP_H_
