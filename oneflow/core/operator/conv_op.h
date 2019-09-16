#ifndef ONEFLOW_CORE_OPERATOR_CONV_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/device/cudnn_conv_ctx_cache.h"

namespace oneflow {

#ifdef WITH_CUDA
class CudnnConvDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvDesc);
  CudnnConvDesc() = delete;
  ~CudnnConvDesc();

  CudnnConvDesc(const DataType& data_type, const Shape& in_blob_shape, const PbMessage& conv_conf);

  const cudnnConvolutionDescriptor_t& Get() const { return val_; }

 private:
  cudnnConvolutionDescriptor_t val_;
};
#endif  // WITH_CUDA

struct ConvOpCtx : public OpContext {
  int64_t col_buf_size;
#ifdef WITH_CUDA
  CudnnConvAlgoCtx cudnn_conv_algo_ctx;
#endif  // WITH_CUDA
};

template<int32_t NDims>
class ConvOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvOp);
  ConvOp() = default;
  virtual ~ConvOp() = default;

  void InitFromOpConf() override;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature,
                                std::function<void(OpContext*)> EnrollOpCtx) const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
  PbMessage* MutableCustomizedKernelConf(KernelConf*) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*, const OpContext*) const override;
  void GenKernelConfWithoutCudnn(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      ConvKernelConf* conv_conf) const;
  void GenKernelConfWithCudnn(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              KernelConf* kernel_conf, ConvKernelConf* conv_conf,
                              const OpContext*) const;
#ifdef WITH_CUDA
  void InferCudnnAlgo(std::function<const BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
                      CudnnConvAlgoCtx* conv_ctx) const;
#endif  // WITH_CUDA
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_OP_H_
