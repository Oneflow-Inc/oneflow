#include "oneflow/xrt/openvino/ops/op_context.h"
#include "oneflow/xrt/openvino/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace openvino {

class ArgumentOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext *ctx) override {}
};

REGISTER_OPENVINO_OP_KERNEL(Argument, ArgumentOp).Finalize();

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
