#ifndef ONEFLOW_XRT_OPENVINO_OPS_OP_KERNEL_H_
#define ONEFLOW_XRT_OPENVINO_OPS_OP_KERNEL_H_

#include "oneflow/xrt/kernel/op_kernel.h"
#include "oneflow/xrt/openvino/ops/op_context.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/registry.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {
namespace openvino {

class OpenvinoOpKernel : public OpKernel<OpenvinoOpContext> {
 public:
  virtual void Compile(OpenvinoOpContext *ctx) = 0;

  OpenvinoOpKernel() = default;
  virtual ~OpenvinoOpKernel() = default;
};

using OpenvinoOpKernelPtr = std::shared_ptr<OpKernel<OpenvinoOpContext>>;

#define REGISTER_OPENVINO_OP_KERNEL(OpName, KernelType)                       \
  static OpKernelRegistrar<OpenvinoOpContext> _openvino_op_kernel_##OpName##_ \
      __attribute__((unused)) =                                               \
          OpKernelRegistrar<OpenvinoOpContext>(#OpName)                       \
              .SetField(XrtEngine::OPENVINO)                                  \
              .SetDevice({XrtDevice::CPU_X86})                                \
              .SetFactory([]() -> OpKernel<OpenvinoOpContext> * { return new KernelType; })

inline OpenvinoOpKernelPtr BuildOpKernel(const std::string &op_name) {
  auto field = MakeXrtField(XrtDevice::CPU_X86, XrtEngine::OPENVINO);
  return OpenvinoOpKernelPtr(OpKernelBuilder<OpenvinoOpContext>()(field, op_name));
}

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_OPENVINO_OPS_OP_KERNEL_H_
