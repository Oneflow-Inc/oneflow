#ifndef ONEFLOW_CORE_OPERATOR_CONVOLUTION_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONVOLUTION_OP_H_

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

#ifdef WITH_CUDNN
class CudnnConvolutionOpUtil {
 public:
  CudnnConvolutionOpUtil();
  ~CudnnConvolutionOpUtil();

  void InitTensorDesc(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ConvolutionOpConf& conv_conf);
  cudnnConvolutionFwdAlgo_t InferFwdAlgo();
  cudnnConvolutionBwdFilterAlgo_t InferBwdFilterAlgo();
  cudnnConvolutionBwdDataAlgo_t InferBwdDataAlgo();
  size_t InferWorkspaceSize(
      cudnnConvolutionFwdAlgo_t cudnn_fwd_algo,
      cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo,
      cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo);

 private:
  cudaStream_t cuda_stream_;
  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
};
#endif  // WITH_CUDNN

class ConvolutionOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionOp);
  ConvolutionOp() = default;
  ~ConvolutionOp() = default;

  void InitFromOpConf() override;

  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }
  const PbMessage& GetSpecialConf() const override;
  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

  int32_t ModelSplitAxis() const override { return 1; }
  int32_t MaxModelSplitNum() const override {
    return op_conf().convolution_conf().out_num();
  }
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONVOLUTION_OP_H_
