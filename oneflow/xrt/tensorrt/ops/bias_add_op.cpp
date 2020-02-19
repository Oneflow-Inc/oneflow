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

    nvinfer1::ITensor *in = ctx->Input("a");

    // std::vector<int64_t> dims(in_shape.NumAxes(), 1);
    // dims[1] = bias_shape.At(0);
    //
    // nvinfer1::ITensor *bias = ctx->Input("b");
    // nvinfer1::ITensor *reshaped_bias =  // NOLINT
    //     helpers::Reshape(ctx, bias, AsShape(dims));
    // // Add bias to input by ElementWise layer.
    // auto *layer = ctx->builder()->addElementWise(  // NOLINT
    //     *in, *reshaped_bias, nvinfer1::ElementWiseOperation::kSUM);

    // TensorRT Scale layer requires 4-d input.
    if (in_shape.NumAxes() != 4) {
      CHECK_LE(in_shape.NumAxes(), 4);
      std::vector<int64_t> dims(4, 1);
      for (int i = 0; i < in_shape.NumAxes(); ++i) { dims[i] = in_shape.At(i); }
      in = helpers::Reshape(ctx, in, AsShape(dims));
    }

    nvinfer1::Weights bias = ctx->Weight("b");
    nvinfer1::Weights gamma{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};

    nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kCHANNEL;
    // out = gamma * (in + bias)
    nvinfer1::IScaleLayer *layer =  // NOLINT
        ctx->builder()->addScale(*in, mode, bias, gamma, power);
    layer->setName(ctx->op_name().c_str());

    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(BiasAdd, BiasAddOp).Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
