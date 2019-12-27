#include "absl/strings/str_cat.h"
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class ConcatOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    int num_inputs = ctx->num_inputs();
    CHECK_GE(num_inputs, 2) << "Concat needs 2 inputs at least.";
    Shape in_shape = ctx->InputShape("in_0");
    int32_t axis = ctx->GetAttr<int32_t>("axis");
    if (axis < 0) { axis += in_shape.NumAxes(); }
    CHECK_GE(axis, 0);
    CHECK_LT(axis, in_shape.NumAxes());

    std::vector<nvinfer1::ITensor *> in(num_inputs);
    for (int i = 0; i < num_inputs; ++i) { in[i] = ctx->Input(absl::StrCat("in_", i)); }
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
