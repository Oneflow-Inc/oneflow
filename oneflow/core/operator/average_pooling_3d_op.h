#ifndef ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_3D_OP_H_
#define ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_3D_OP_H_

#include "oneflow/core/operator/pooling_3d_op.h"

namespace oneflow {

class AveragePooling3DOp final : public Pooling3DOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePooling3DOp);
  AveragePooling3DOp() = default;
  ~AveragePooling3DOp() = default;

  const PbMessage& GetSpecialConf() const override;

 private:
  void VirtualEnrollDataTmpBn() override {}
  void VirtualInferDataTmpBlobDesc(std::function<BlobDesc*(const std::string)>
                                       GetBlobDesc4BnInOp) const override {}
  Pooling3DKernelConf* GetMutPooling3DKernelConf(KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_AVERAGE_POOLING_3D_OP_H_
