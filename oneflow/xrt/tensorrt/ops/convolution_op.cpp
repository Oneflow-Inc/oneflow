#include "NvInfer.h"
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class ConvolutionOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    nvinfer1::ITensor *in = ctx->Input("in");
    nvinfer1::Weights weight = ctx->Weight("weight");

    nvinfer1::Weights bias;
    if (ctx->GetAttr<bool>("use_bias")) {
      bias = ctx->Weight("bias");
    } else {
      bias = nvinfer1::Weights{nvinfer1::DataType::kFLOAT /* type */, nullptr /* values */,
                               0 /* count */};
    }

    CHECK_EQ(ctx->GetAttr<std::string>("data_format"), "channels_first");
    std::vector<int32_t> kernel_size = ctx->GetAttr<std::vector<int32_t>>("kernel_size");
    std::vector<int32_t> strides = ctx->GetAttr<std::vector<int32_t>>("strides");
    std::vector<int32_t> dilation = ctx->GetAttr<std::vector<int32_t>>("dilation_rate");

    int filters = ctx->GetAttr<int32_t>("filters");
    auto *layer = ctx->builder()->addConvolution(
        *in, filters, nvinfer1::DimsHW(kernel_size[0], kernel_size[1]), weight, bias);
    layer->setName(ctx->op_name().c_str());

    layer->setStride(nvinfer1::DimsHW(strides[0], strides[1]));
    layer->setDilation(nvinfer1::DimsHW(dilation[0], dilation[1]));
    // The default padding mode is valid for TensorRT.
    if (ctx->GetAttr<std::string>("padding") == "same") {
      layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_LOWER);
    }
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Conv2D, ConvolutionOp).Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
