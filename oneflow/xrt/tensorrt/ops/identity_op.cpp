#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class IdentityOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    nvinfer1::ITensor *in = ctx->SoleInput();
    auto *layer = ctx->builder()->addIdentity(*in);
    layer->setName(ctx->op_name().c_str());
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Identity, IdentityOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
