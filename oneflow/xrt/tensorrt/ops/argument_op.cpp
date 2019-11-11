#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class ArgumentOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    // nvinfer1::ITensor *value = ctx->Variable("value");
    // ctx->SetOutput("value", value);
  }
};

REGISTER_TRT_OP_KERNEL(Argument, ArgumentOp).Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
