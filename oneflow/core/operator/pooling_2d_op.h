#ifndef ONEFLOW_CORE_OPERATOR_POOLING_2D_OP_H_
#define ONEFLOW_CORE_OPERATOR_POOLING_2D_OP_H_

#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

class Pooling2DOp : public PoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling2DOp);
  Pooling2DOp() = default;
  virtual ~Pooling2DOp() = default;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 protected:
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;
  void VirtualCheckPoolSizeAndStrides() const override;
  int32_t GetPoolSizeH() const override;
  int32_t GetPoolSizeW() const override;
  int32_t GetStridesH() const override;
  int32_t GetStridesW() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_POOLING_2D_OP_H_
