#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"
#include "oneflow/xrt/tensorrt/trt_logger.h"

#include "NvInfer.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template<nvinfer1::PoolingType pooling_type>
class PoolingOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in");
    CHECK_GE(in_shape.NumAxes(), 3);
    CHECK_LE(in_shape.NumAxes(), 5);

    const std::string padding = ctx->GetAttr<std::string>("padding");
    std::vector<int32_t> pool_size = ctx->GetAttr<std::vector<int32_t>>("pool_size");
    std::vector<int32_t> strides = ctx->GetAttr<std::vector<int32_t>>("strides");

    nvinfer1::ITensor *in = ctx->Input("in");
    auto *layer =
        ctx->builder()->addPooling(*in, pooling_type, nvinfer1::DimsHW(pool_size[0], pool_size[1]));
    layer->setName(ctx->op_name().c_str());

    layer->setStride(nvinfer1::DimsHW(strides[0], strides[1]));
    if (padding == "same") { layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_LOWER); }
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(MaxPooling2D, PoolingOp<nvinfer1::PoolingType::kMAX>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(AveragePooling2D, PoolingOp<nvinfer1::PoolingType::kAVERAGE>)
    .EnableTrainPhase()
    .Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
