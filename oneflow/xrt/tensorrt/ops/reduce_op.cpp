#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class ReduceOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in");

    std::vector<int32_t> axis = ctx->GetAttr<std::vector<int32_t>>("axis");
    int32_t reduceAxes = 0;
    for(int i = 0; i < axis.size(); ++i) {
      reduceAxes = reduceAxes | (1U << axis[i]);
    } 
    bool keepDimensions = ctx->GetAttr<bool>("keep_dims"); 
    nvinfer1::ITensor *in = ctx->Input("in");
    auto *layer = ctx->builder()->addReduce(*in, nvinfer1::ReduceOperation::kSUM, 
        reduceAxes, keepDimensions);
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(ReduceSum, ReduceOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
