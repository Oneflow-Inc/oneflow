#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"
#include "absl/strings/str_cat.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class AddOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in_0");
    nvinfer1::ITensor *in_0 = ctx->Input("in_0");
    nvinfer1::ITensor *sum = in_0;
  
    int num_inputs = ctx->num_inputs();
    
    auto *layer = ctx->builder()->addElementWise(*in_0, *sum, nvinfer1::ElementWiseOperation::kSUM);
    for(int i = 1; i < num_inputs; ++i) {
      std::string name = absl::StrCat("in_", i);
      CHECK_EQ(in_shape, ctx->InputShape(name));
      layer = ctx->builder()->addElementWise(*ctx->Input(name), *sum, nvinfer1::ElementWiseOperation::kSUM);  
      sum =  layer->getOutput(0);
    }
    
    ctx->SetOutput("out", sum);
  }
};

REGISTER_TRT_OP_KERNEL(Add, AddOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
