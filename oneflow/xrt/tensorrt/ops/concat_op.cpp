#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/xrt/tensorrt/trt_logger.h"

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "strings.h"
namespace oneflow {
namespace xrt {
namespace tensorrt {

class ConcatOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in");

    nvinfer1::ITensor* const in = ctx->Input("in");
    int num_inputs = ctx->num_inputs();
    int32_t axis = ctx->GetAttr<int32_t>("axis");
    auto *layer = ctx->builder()->addConcatenation(&in, num_inputs);
    layer->setAxis(axis);
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Concat, ConcatOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
