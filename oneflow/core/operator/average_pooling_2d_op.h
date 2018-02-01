#ifndef ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_2D_OP_H_
#define ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_2D_OP_H_

#include "oneflow/core/operator/pooling_2d_op.h"

namespace oneflow {

class AveragePooling2DOp final : public Pooling2DOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePooling2DOp);
  AveragePooling2DOp() = default;
  ~AveragePooling2DOp() = default;

  const PbMessage& GetSpecialConf() const override;

 private:
  PoolingKernelConf* GetMutPoolingKernelConf(KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_2D_OP_H_
