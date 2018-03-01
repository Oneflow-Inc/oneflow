#ifndef ONEFLOW_CORE_OPERATOR_CONV_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

class CudnnConvDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvDesc);
  CudnnConvDesc() = delete;
  ~CudnnConvDesc();

  CudnnConvDesc(const BlobDesc*, const int, const std::vector<int>&,
                const std::vector<int>&, const std::vector<int>&,
                const std::string&, const std::string&);

  const cudnnConvolutionDescriptor_t& Get() const { return val_; }

 private:
  cudnnConvolutionDescriptor_t val_;
};

class ConvOpCtx final : public OpContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvOpCtx);
  ConvOpCtx() = default;
  ~ConvOpCtx() = default;

  void set_cudnn_fwd_algo(int32_t cudnn_fwd_algo) {
    cudnn_fwd_algo_ = cudnn_fwd_algo;
  }
  void set_cudnn_bwd_filter_algo(int32_t cudnn_bwd_filter_algo) {
    cudnn_bwd_filter_algo_ = cudnn_bwd_filter_algo;
  }
  void set_cudnn_bwd_data_algo(int32_t cudnn_bwd_data_algo) {
    cudnn_bwd_data_algo_ = cudnn_bwd_data_algo;
  }
  const int32_t& cudnn_fwd_algo() const { return cudnn_fwd_algo_; }
  const int32_t& cudnn_bwd_filter_algo() const {
    return cudnn_bwd_filter_algo_;
  }
  const int32_t& cudnn_bwd_data_algo() const { return cudnn_bwd_data_algo_; }

 private:
  int32_t cudnn_fwd_algo_;
  int32_t cudnn_bwd_filter_algo_;
  int32_t cudnn_bwd_data_algo_;
};

class ConvOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvOp);
  ConvOp() = default;
  ~ConvOp() = default;

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
    return kernel_conf->mutable_conv_3d_conf();
  }
  virtual int32_t KernelDimSize() const = 0;
  size_t InferCudnnWorkspaceSize(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      std::function<void(OpContext*)> EnrollOpContext) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_OP_H_
