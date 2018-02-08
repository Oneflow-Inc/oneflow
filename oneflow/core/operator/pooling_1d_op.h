#ifndef ONEFLOW_CORE_OPERATOR_POOLING_1D_OP_H_
#define ONEFLOW_CORE_OPERATOR_POOLING_1D_OP_H_

#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

class Pooling1DOp : public PoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling1DOp);
  Pooling1DOp() = default;
  virtual ~Pooling1DOp() = default;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 protected:
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;
  void VirtualCheckPoolSizeAndStrides() const override;
  int32_t GetPoolSizeLength() const;
  int32_t GetStridesLength() const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_POOLING_1D_OP_H_
