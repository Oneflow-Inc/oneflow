#ifndef ONEFLOW_CORE_OPERATOR_DECONV_OP_H_
#define ONEFLOW_CORE_OPERATOR_DECONV_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/device/cudnn_conv_ctx_cache.h"

namespace oneflow {

#ifdef WITH_CUDA

class CudnnDeconvDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnDeconvDesc);
  CudnnDeconvDesc() = delete;
  ~CudnnDeconvDesc();

  CudnnDeconvDesc(const DataType&, const ShapeView&, const ConvConf&);

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

}  //  namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DECONV_OP_H_
