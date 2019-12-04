#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class GatherOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in");
    //    CHECK_GE(in_shape.NumAxes(), 2);

    int axis = ctx->GetAttr<int64_t>("axis");
    // CHECK_GE(axis, 1);
    // CHECK_LT(axis, in_shape.NumAxes());
    nvinfer1::ITensor *indices = ctx->Input("indices");
    nvinfer1::ITensor *in = ctx->Input("in");
    auto *layer = ctx->builder()->addGather(*in, *indices, axis);
    layer->setNbElementWiseDims(0);
    layer->setName(ctx->op_name().c_str());
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Gather, GatherOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
