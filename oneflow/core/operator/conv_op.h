#ifndef ONEFLOW_CORE_OPERATOR_CONV_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

#ifdef WITH_CUDA
struct CudnnConvAlgoCtx final {
  cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf;
  cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_algo_perf;
  cudnnConvolutionBwdDataAlgoPerf_t bwd_data_algo_perf;
};

class CudnnConvDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvDesc);
  CudnnConvDesc() = delete;
  ~CudnnConvDesc();

  CudnnConvDesc(const DataType& data_type, const Shape& in_blob_shape,
                const int kernel_dim, const int* dilation_rate,
                const int* strides, const int* kernel_size,
                const std::string& data_format, const std::string& padding);

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

  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }
  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext*, DeviceType) const override;

  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*, KernelConf*) const override;

 protected:
  PbMessage* MutableCustomizedKernelConf(KernelConf*) const override;
  int32_t ModelSplitAxis() const override;
  int32_t MaxModelSplitNum() const override;

#ifdef WITH_CUDA
  void InferCudnnAlgo(
      std::function<const BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      CudnnConvAlgoCtx* conv_ctx) const;
#endif  // WITH_CUDA
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_OP_H_
