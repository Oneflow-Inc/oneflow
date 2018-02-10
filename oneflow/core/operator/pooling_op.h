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
  int32_t GetDInPbRf(const std::string& field_name) const;
  int32_t GetHInPbRf(const std::string& field_name) const;
  int32_t GetWInPbRf(const std::string& field_name) const;
  int32_t GetInD(const Shape& in_shape) const;
  int32_t GetInH(const Shape& in_shape) const;
  int32_t GetInW(const Shape& in_shape) const;
  std::tuple<int32_t, int32_t> CalcOutDAndPaddingD(const Shape& in_shape) const;
  std::tuple<int32_t, int32_t> CalcOutHAndPaddingH(const Shape& in_shape) const;
  std::tuple<int32_t, int32_t> CalcOutWAndPaddingW(const Shape& in_shape) const;
  virtual int32_t GetDim() const = 0;
  void CheckPoolSizeAndStrides() const;
  Shape CalcOutShape(int32_t in_n, int32_t in_c, int32_t out_d, int32_t out_h,
                     int32_t out_w) const;
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_POOLING_OP_H_
