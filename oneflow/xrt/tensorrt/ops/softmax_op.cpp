#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class SoftmaxOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in");
    CHECK_GE(in_shape.NumAxes(), 2);

    int32_t axis = ctx->GetAttr<int32_t>("axis");
    if (axis < 0) { axis += in_shape.NumAxes(); }
    CHECK_LT(axis, in_shape.NumAxes());

    nvinfer1::ITensor *in = ctx->Input("in");
    auto *layer = ctx->builder()->addSoftMax(*in);
    layer->setAxes((1U << axis));
    layer->setName(ctx->op_name().c_str());
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Softmax, SoftmaxOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
