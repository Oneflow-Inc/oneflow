#ifndef ONEFLOW_CORE_OPERATOR_CONV_V2_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_V2_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/device/cudnn_conv_ctx_cache.h"
#include "oneflow/core/operator/conv_op.h"

namespace oneflow {

template<int32_t NDims>
class ConvV2Op : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvV2Op);
  ConvV2Op() = default;
  virtual ~ConvV2Op() = default;

  void InitFromOpConf() override;
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx, int64_t record_piece_size,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;
  void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext*, const OpContext*) const override;

  int32_t OutputBlobModelSplitAxis(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const std::string& obn) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }

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
                      CudnnConvAlgoCtx* conv_ctx, const int64_t device_id) const;
#endif  // WITH_CUDA
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_V2_OP_H_
