#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

#include "oneflow/xrt/api.h"
#include "oneflow/xrt/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class BiasAddOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    CHECK_EQ(ctx->InputType("a"), ctx->InputType("b"));

    Shape in_shape = ctx->InputShape("a");
    Shape bias_shape = ctx->InputShape("b");
    CHECK_GE(in_shape.NumAxes(), 2);
    CHECK_EQ(bias_shape.NumAxes(), 1);

    std::vector<int64_t> dims(in_shape.NumAxes(), 1);
    dims[1] = bias_shape.At(0);

    nvinfer1::ITensor *in = ctx->Input("a");
    nvinfer1::ITensor *bias = ctx->Input("b");
    // nvinfer1::Weights bias = ctx->Weight("b");
    nvinfer1::ITensor *reshaped_bias = helpers::Reshape(ctx, bias, AsShape(dims));
    // Add bias to input by ElementWise layer.
    auto *layer = ctx->builder()->addElementWise(  // NOLINT
        *in, *reshaped_bias, nvinfer1::ElementWiseOperation::kSUM);
    layer->setName(ctx->op_name().c_str());

    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(BiasAdd, BiasAddOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
