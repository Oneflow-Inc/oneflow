#ifndef ONEFLOW_CORE_OPERATOR_MAX_POOLING_2D_OP_H_
#define ONEFLOW_CORE_OPERATOR_MAX_POOLING_2D_OP_H_

#include "oneflow/core/operator/pooling_2d_op.h"

namespace oneflow {

class MaxPooling2DOp final : public Pooling2DOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling2DOp);
  MaxPooling2DOp() = default;
  ~MaxPooling2DOp() = default;

  const PbMessage& GetSpecialConf() const override;

 private:
  Pooling3DKernelConf* GetMutPooling3DKernelConf(KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MAX_POOLING_2D_OP_H_
