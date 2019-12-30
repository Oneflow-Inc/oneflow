#include "NvInfer.h"
#include "absl/strings/str_cat.h"
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template<nvinfer1::ElementWiseOperation element_wise_op>
class ElementWiseOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    int num_inputs = ctx->num_inputs();
    CHECK_GE(num_inputs, 2) << "ElementWiseOp needs 2 inputs at least.";

    Shape in_shape = ctx->InputShape("in_0");
    nvinfer1::ITensor *result = ctx->Input("in_0");
    for (int i = 1; i < num_inputs; ++i) {
      std::string name = absl::StrCat("in_", i);
      CHECK_EQ(in_shape, ctx->InputShape(name));
      auto *layer = ctx->builder()->addElementWise(*ctx->Input(name), *result, element_wise_op);
      result = layer->getOutput(0);
    }
    ctx->SetOutput("out", result);
  }
};

REGISTER_TRT_OP_KERNEL(Add, ElementWiseOp<nvinfer1::ElementWiseOperation::kSUM>)
    .EnableTrainPhase()
    .Finalize();

REGISTER_TRT_OP_KERNEL(Multiply, ElementWiseOp<nvinfer1::ElementWiseOperation::kPROD>)
    .EnableTrainPhase()
    .Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
