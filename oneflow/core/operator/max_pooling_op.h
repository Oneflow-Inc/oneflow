#ifndef ONEFLOW_CORE_OPERATOR_MAX_POOLING_OP_H_
#define ONEFLOW_CORE_OPERATOR_MAX_POOLING_OP_H_

#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

class MaxPoolingOp final : public PoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingOp);
  MaxPoolingOp() = default;
  ~MaxPoolingOp() = default;

  const PbMessage& GetSpecialConf() const override;

 private:
  void VirtualEnrollDataTmpBn() override;
  void VirtualInferDataTmpBlobDesc(std::function<BlobDesc*(const std::string)>
                                       GetBlobDesc4BnInOp) const override;
  PoolingKernelConf* GetMutPoolingKernelConf(KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MAX_POOLING_OP_H_
