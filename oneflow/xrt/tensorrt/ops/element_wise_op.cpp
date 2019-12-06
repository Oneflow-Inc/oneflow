#include "NvInfer.h"
#include "absl/strings/str_cat.h"
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template <nvinfer1::ElementWiseOperation element_wise_operation>
class ElementWiseOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in_0");
    nvinfer1::ITensor *in_0 = ctx->Input("in_0");
    nvinfer1::ITensor *tmp_result = in_0;

    int num_inputs = ctx->num_inputs();

    auto *layer = ctx->builder()->addElementWise(
        *in_0, *tmp_result, nvinfer1::ElementWiseOperation::kSUM);
    for (int i = 1; i < num_inputs; ++i) {
      std::string name = absl::StrCat("in_", i);
      CHECK_EQ(in_shape, ctx->InputShape(name));
      layer = ctx->builder()->addElementWise(*ctx->Input(name), *tmp_result,
                                             element_wise_operation);
      tmp_result = layer->getOutput(0);
    }
    ctx->SetOutput("out", tmp_result);
  }
};

REGISTER_TRT_OP_KERNEL(Add, ElementWiseOp<nvinfer1::ElementWiseOperation::kSUM>)
    .EnableTrainPhase()
    .Finalize();

REGISTER_TRT_OP_KERNEL(Multiply,
                       ElementWiseOp<nvinfer1::ElementWiseOperation::kPROD>)
    .EnableTrainPhase()
    .Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
