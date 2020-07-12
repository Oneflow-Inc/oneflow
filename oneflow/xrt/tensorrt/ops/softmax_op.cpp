#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class SoftmaxOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->SoleInputShape();
    CHECK_GE(in_shape.NumAxes(), 2);
    int32_t axis = in_shape.NumAxes() - 1;
    nvinfer1::ITensor *in = ctx->SoleInput();
    auto *layer = ctx->builder()->addSoftMax(*in);
    layer->setAxes((1U << axis));
    layer->setName(ctx->op_name().c_str());
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Softmax, SoftmaxOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
