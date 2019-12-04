#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template <nvinfer1::ReduceOperation reduce_operation>
class ReduceOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    std::vector<int32_t> axis = ctx->GetAttr<std::vector<int32_t>>("axis");
    int32_t reduceAxes = 0;
    for (int i = 0; i < axis.size(); ++i) {
      reduceAxes = reduceAxes | (1U << axis[i]);
    }
    bool keepDimensions = ctx->GetAttr<bool>("keep_dims");
    nvinfer1::ITensor *in = ctx->Input("in");
    auto *layer = ctx->builder()->addReduce(*in, reduce_operation, reduceAxes,
                                            keepDimensions);
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
