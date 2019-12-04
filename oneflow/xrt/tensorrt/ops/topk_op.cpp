#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class TopKOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in");
    //    CHECK_GE(in_shape.NumAxes(), 2);

    int32_t k = ctx->GetAttr<int32_t>("k");
    // CHECK_GE(axis, 1);
    // CHECK_LT(axis, in_shape.NumAxes());
    uint32_t reduceAxes = (1U << in_shape.dim_vec().back());
    nvinfer1::ITensor *in = ctx->Input("in");
    auto *layer = ctx->builder()->addTopK(*in, nvinfer1::TopKOperation::kMAX, k,
                                          reduceAxes);
    layer->setName(ctx->op_name().c_str());
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(TopK, TopKOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
