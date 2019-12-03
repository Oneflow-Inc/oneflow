#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/xrt/tensorrt/trt_logger.h"

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "absl/strings/str_cat.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class ConcatOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in_0"); 
    std::vector<nvinfer1::ITensor*> in;

    int num_inputs = ctx->num_inputs();
        int32_t axis = ctx->GetAttr<int32_t>("axis");
        for(int i = 0; i < num_inputs; ++i) {
      std::string name = absl::StrCat("in_", i);
      in.push_back(ctx->Input(name));
    }
    auto *layer = ctx->builder()->addConcatenation(in.data(), num_inputs);
    layer->setAxis(axis);
    layer->setName(ctx->op_name().c_str());
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Concat, ConcatOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
