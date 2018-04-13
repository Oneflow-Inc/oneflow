#ifndef ONEFLOW_CORE_OPERATOR_CONV_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

#ifdef WITH_CUDA
struct CudnnConvAlgoCtx final {
  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  size_t fwd_ws_size;
  size_t bwd_filter_ws_size;
  size_t bwd_data_ws_size;
};

class CudnnConvDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvDesc);
  CudnnConvDesc() = delete;
  ~CudnnConvDesc();

  CudnnConvDesc(const DataType& data_type, const Shape& in_blob_shape,
                const PbMessage& conv_conf);

  const cudnnConvolutionDescriptor_t& Get() const { return val_; }

 private:
  cudnnConvolutionDescriptor_t val_;
};
#endif  // WITH_CUDA

template<int32_t NDims>
class ConvOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvOp);
  ConvOp() = default;
  virtual ~ConvOp() = default;

  void InitFromOpConf() override;
  bool NeedOutWhenBackward() const override;
  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext*, DeviceType) const override;

  int32_t ModelSplitAxis() const override;
  int32_t MaxModelSplitNum() const override;

 private:
  PbMessage* MutableCustomizedKernelConf(KernelConf*) const override;
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*, KernelConf*) const override;
  void GenKernelConfWithoutCudnn(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      ConvKernelConf* conv_conf) const;
  void GenKernelConfWithCudnn(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      KernelConf* kernel_conf, ConvKernelConf* conv_conf) const;
#ifdef WITH_CUDA
  void InferCudnnAlgo(
      std::function<const BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      CudnnConvAlgoCtx* conv_ctx) const;
#endif  // WITH_CUDA
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_OP_H_
