#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template<nvinfer1::ReduceOperation reduce_op>
class ReduceOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");

    int32_t reduce_axis = 0;
    for (int i = 0; i < axis.size(); ++i) { reduce_axis = reduce_axis | (1U << axis[i]); }
    bool keepDimensions = ctx->Attr<bool>("keep_dims");
    // TensorRT does not support full reduce without keepDimensions.
    Shape in_shape = ctx->InputShape("in");
    if (!keepDimensions) {
      CHECK_NE(reduce_axis, (1U << in_shape.NumAxes()) - 1)
          << "TensorRT does not support full reduce without keepDimensions.";
    }

    nvinfer1::ITensor *in = ctx->Input("in");
    auto *layer = ctx->builder()->addReduce(*in, reduce_op, reduce_axis, keepDimensions);
    layer->setName(ctx->op_name().c_str());
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(ReduceSum, ReduceOp<nvinfer1::ReduceOperation::kSUM>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(ReduceMean, ReduceOp<nvinfer1::ReduceOperation::kAVG>)
    .EnableTrainPhase()
    .Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
