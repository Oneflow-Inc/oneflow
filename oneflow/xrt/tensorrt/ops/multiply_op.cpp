#include "NvInfer.h"
#include "absl/strings/str_cat.h"
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class MultiplyOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape x_shape = ctx->InputShape("x_0");
    Shape y_shape = ctx->InputShape("y_0");
    nvinfer1::ITensor *x = ctx->Input("x_0");
    nvinfer1::ITensor *y = ctx->Input("y_0");
    CHECK_EQ(x_shape, y_shape);
    auto *layer = ctx->builder()->addElementWise(*x, *y, nvinfer1::ElementWiseOperation::kPROD); 
    ctx->SetSoleOutput(layer->getOutput(0));  
  }
};

REGISTER_TRT_OP_KERNEL(Multiply, MultiplyOp)
    .EnableTrainPhase()
    .Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
