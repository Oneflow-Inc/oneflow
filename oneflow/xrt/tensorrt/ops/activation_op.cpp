#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/xrt/tensorrt/trt_logger.h"
#include "NvInferRuntimeCommon.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

nvinfer1::ActivationType GetActivationType() {
  if (oneflow::ActivationType::kRelu) 
    //case oneflow::kRelu:
      return nvinfer1::ActivationType::kRELU;
   else if (oneflow::ActivationType::kTanH)      
      return nvinfer1::ActivationType::kTANH;
    else if (oneflow::ActivationType::kSigmoid)     
      return nvinfer1::ActivationType::kSIGMOID; 
    else {
      LOG(FATAL) << "No valid activation type get!"; 
      return nvinfer1::ActivationType::kRELU;
    } 
  
}

class ActivationOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in");

    nvinfer1::ITensor *in = ctx->Input("in");
    auto *layer = ctx->builder()->addActivation(*in, GetActivationType());
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Activation, ActivationOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
