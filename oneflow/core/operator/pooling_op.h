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

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 protected:
  virtual void VirtualEnrollDataTmpBn() = 0;
  virtual void VirtualInferDataTmpBlobDesc(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp) const = 0;
  virtual Pooling3DKernelConf* GetMutPooling3DKernelConf(KernelConf*) const = 0;
  virtual int32_t GetDim() const = 0;
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;

 private:
  int64_t GetTensorDimInOpConf(const std::string& field_name,
                               uint8_t dim) const;
  std::vector<int64_t> GetTensorInOpConf(const std::string& field_name) const;
  int64_t GetInDim(const Shape& in_shape, uint8_t dim) const;
  void CheckPoolSizeAndStrides() const;
  Shape GetOutShape(int64_t in_n, int64_t in_c,
                    const std::vector<int64_t>& out) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_POOLING_OP_H_
