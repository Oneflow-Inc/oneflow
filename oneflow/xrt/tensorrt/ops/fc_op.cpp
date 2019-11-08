#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class FullyConnectedOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    nvinfer1::ITensor *in = ctx->Input("in");
    // Transpose weight
    nvinfer1::Weights weight = ctx->Weight("weight");

    nvinfer1::Weights bias;
    if (ctx->GetAttr<bool>("use_bias")) {
      bias = ctx->Weight("bias");
    }

    auto *layer = ctx->addFullyConnected(*in, 1, weight, bias);
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(FullyConnected, FullyConnectedOp);

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
