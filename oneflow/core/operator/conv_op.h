#ifndef ONEFLOW_CORE_OPERATOR_CONV_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

#ifdef WITH_CUDA
class CudnnConvDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvDesc);
  CudnnConvDesc() = delete;
  ~CudnnConvDesc();

  CudnnConvDesc(const DataType& data_type, const Shape& in_blob_shape,
                const int kernel_dim_size, const int* dilation_rate,
                const int* strides, const int* kernel_size,
                const std::string& data_format, const std::string& padding);

  const cudnnConvolutionDescriptor_t& Get() const { return val_; }

 private:
  cudnnConvolutionDescriptor_t val_;
};
#endif  // WITH_CUDA

class ConvOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvOp);
  ConvOp() = default;
  virtual ~ConvOp() = default;

  void InitFromOpConf() override;

  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }
  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, DeviceType device_type,
      std::function<void(OpContext*)> EnrollOpContext) const override;

  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, const OpContext* op_ctx,
      KernelConf* kernel_conf) const override;

 protected:
  PbMessage* MutableCustomizedKernelConf(
      KernelConf* kernel_conf) const override {
    return kernel_conf->mutable_conv_conf();
  }
  virtual int32_t KernelDimSize() const = 0;
#ifdef WITH_CUDA
  size_t InferCudnnWorkspaceSize(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      std::function<void(OpContext*)> EnrollOpContext) const;
#endif  // WITH_CUDA
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_OP_H_
