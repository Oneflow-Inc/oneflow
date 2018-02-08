#ifndef ONEFLOW_CORE_OPERATOR_POOLING_3D_OP_H_
#define ONEFLOW_CORE_OPERATOR_POOLING_3D_OP_H_

#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

class Pooling3DOp : public PoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling3DOp);
  Pooling3DOp() = default;
  virtual ~Pooling3DOp() = default;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 protected:
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;
  void VirtualCheckPoolSizeAndStrides() const override;
  int32_t GetPoolSizeD() const;
  int32_t GetPoolSizeH() const;
  int32_t GetPoolSizeW() const;
  int32_t GetStridesD() const;
  int32_t GetStridesH() const;
  int32_t GetStridesW() const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_POOLING_3D_OP_H_
