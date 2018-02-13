#ifndef ONEFLOW_CORE_OPERATOR_MAX_POOLING_3D_OP_H_
#define ONEFLOW_CORE_OPERATOR_MAX_POOLING_3D_OP_H_

#include "oneflow/core/operator/pooling_3d_op.h"

namespace oneflow {

class MaxPooling3DOp final : public Pooling3DOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling3DOp);
  MaxPooling3DOp() = default;
  ~MaxPooling3DOp() = default;

  const PbMessage& GetSpecialConf() const override;

 private:
  Pooling3DKernelConf* GetMutPooling3DKernelConf(KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MAX_POOLING_3D_OP_H_
