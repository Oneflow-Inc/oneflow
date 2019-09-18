#ifndef ONEFLOW_CORE_OPERATOR_DECONV_OP_H_
#define ONEFLOW_CORE_OPERATOR_DECONV_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/device/cudnn_conv_ctx_cache.h"

namespace oneflow {

#ifdef WITH_CUDA

class CudnnDeconvDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnDeconvDesc);
  CudnnDeconvDesc() = delete;
  ~CudnnDeconvDesc();

  CudnnDeconvDesc(const DataType& data_type, const Shape& in_blob_shape,
                  const PbMessage& deconv_conf);

  const cudnnConvolutionDescriptor_t& Get() const { return val_; }

 private:
  cudnnConvolutionDescriptor_t val_;
};
#endif  //  WITH_CUDA

struct DeconvOpCtx : public OpContext {
#ifdef WITH_CUDA
  CudnnConvAlgoCtx cudnn_deconv_algo_ctx;
#endif  //  WITH_CUDA
};

template<int32_t NDims>
class DeconvOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeconvOp);
  DeconvOp() = default;
  virtual ~DeconvOp() = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext*, int64_t record_piece_size,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;
  void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext*, const OpContext*) const override;
  int32_t OutputBlobModelSplitAxis(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const std::string& obn) const override;

 private:
  PbMessage* MutableCustomizedKernelConf(KernelConf*) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInop,
                            const ParallelContext*, KernelConf*, const OpContext*) const override;
  void GenKernelConfWithCudnn(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInop,
                              KernelConf* kernel_conf, DeconvKernelConf* deconv_conf,
                              const OpContext*) const;
#ifdef WITH_CUDA
  void InferCudnnAlgo(std::function<const BlobDesc*(const std::string)> GetBlobDesc4BnInop,
                      CudnnConvAlgoCtx* deconv_ctx, const int64_t device_id) const;
#endif  // WITH_CUDA
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
};

}  //  namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DECONV_OP_H_
