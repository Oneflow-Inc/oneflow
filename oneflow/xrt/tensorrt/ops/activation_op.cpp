#include "NvInferRuntimeCommon.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"
#include "oneflow/xrt/tensorrt/trt_logger.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template <nvinfer1::ActivationType activation_type>
class ActivationOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in");

    nvinfer1::ITensor *in = ctx->Input("in");
    auto *layer = ctx->builder()->addActivation(*in, activation_type);
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Tanh, ActivationOp<nvinfer1::ActivationType::kTANH>).EnableTrainPhase().Finalize();
REGISTER_TRT_OP_KERNEL(Relu, ActivationOp<nvinfer1::ActivationType::kRELU>).EnableTrainPhase().Finalize();
REGISTER_TRT_OP_KERNEL(Sigmoid, ActivationOp<nvinfer1::ActivationType::kSIGMOID>).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
