#ifndef ONEFLOW_CORE_OPERATOR_POOLING_2D_OP_H_
#define ONEFLOW_CORE_OPERATOR_POOLING_2D_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class Pooling2DOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling2DOp);
  Pooling2DOp() = default;
  virtual ~Pooling2DOp() = default;

  void InitFromOpConf() override;

  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 protected:
  virtual void VirtualEnrollDataTmpBn() {}
  virtual void VirtualInferDataTmpBlobDesc(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp) const {}
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;
  virtual Pooling2DKernelConf* GetMutPooling2DKernelConf(KernelConf*) const = 0;

 private:
  std::tuple<int, int> CalcOutSize(int32_t in_h, int32_t in_w) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_POOLING_2D_OP_H_
