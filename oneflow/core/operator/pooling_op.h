#ifndef ONEFLOW_CORE_OPERATOR_POOLING_OP_H_
#define ONEFLOW_CORE_OPERATOR_POOLING_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

class PoolingOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingOp);
  PoolingOp() = default;
  virtual ~PoolingOp() = default;

  void InitFromOpConf() override;

  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }

 protected:
  virtual void VirtualEnrollDataTmpBn() = 0;
  virtual void VirtualInferDataTmpBlobDesc(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp) const = 0;
  virtual void VirtualCheckPoolSizeAndStrides() const = 0;
  virtual Pooling3DKernelConf* GetMutPooling3DKernelConf(KernelConf*) const = 0;
  virtual int32_t GetPoolSizeD() const { UNEXPECTED_RUN(); }
  virtual int32_t GetPoolSizeH() const { UNEXPECTED_RUN(); }
  virtual int32_t GetPoolSizeW() const { UNEXPECTED_RUN(); }
  virtual int32_t GetStridesD() const { UNEXPECTED_RUN(); }
  virtual int32_t GetStridesH() const { UNEXPECTED_RUN(); }
  virtual int32_t GetStridesW() const { UNEXPECTED_RUN(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_POOLING_OP_H_
