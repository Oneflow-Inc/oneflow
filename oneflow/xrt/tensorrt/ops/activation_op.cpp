#include "NvInferRuntimeCommon.h"  // nvinfer1::ActivationType

#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template<nvinfer1::ActivationType activation_type>
class ActivationOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    nvinfer1::ITensor *in = ctx->Input("in");
    auto *layer = ctx->builder()->addActivation(*in, activation_type);
    layer->setName(ctx->op_name().c_str());
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Tanh, ActivationOp<nvinfer1::ActivationType::kTANH>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(Relu, ActivationOp<nvinfer1::ActivationType::kRELU>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(Sigmoid, ActivationOp<nvinfer1::ActivationType::kSIGMOID>)
    .EnableTrainPhase()
    .Finalize();

template<>
class ActivationOp<nvinfer1::ActivationType::kLEAKY_RELU> : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    nvinfer1::ITensor *in = ctx->Input("in");
    auto *layer = ctx->builder()->addActivation(*in, nvinfer1::ActivationType::kLEAKY_RELU);
    layer->setAlpha(ctx->GetAttr<float>("alpha"));
    layer->setName(ctx->op_name().c_str());
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(LeakyRelu, ActivationOp<nvinfer1::ActivationType::kLEAKY_RELU>)
    .EnableTrainPhase()
    .Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
