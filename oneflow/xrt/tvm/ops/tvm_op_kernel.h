#ifndef ONEFLOW_XRT_TVM_OPS_OP_KERNEL_H_
#define ONEFLOW_XRT_TVM_OPS_OP_KERNEL_H_

#include "oneflow/xrt/kernel/op_kernel.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/registry.h"
#include "oneflow/xrt/utility/stl.h"
#include "oneflow/xrt/tvm/ops/tvm_op_context.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

class TVMOpKernel : public OpKernel<TVMOpContext> {
 public:
  virtual void Compile(TVMOpContext* ctx) = 0;

  TVMOpKernel() = default;
  virtual ~TVMOpKernel() = default;
};

#define REGISTER_TVM_OP_KERNEL(OpName, KernelType) \
  static OpKernelRegistrar<TVMOpContext> _tvm_op_kernel_##OpName##_ __attribute__((unused)) = \
  OpKernelRegistrar<TVMOpContext> (#OpName) \
  .SetField(XrtEngine::TVM) \
  .SetDevice({XrtDevice::GPU_CUDA}) \
  .SetFactory([]() -> OpKernel<TVMOpContext>* {return new KernelType; })

using TVMOpKernelPtr = std::shared_ptr<OpKernel<TVMOpContext>>;

inline TVMOpKernelPtr BuildTVMOpKernel(const std::string& op_name) {
  auto field = MakeXrtField(XrtDevice::GPU_CUDA, XrtEngine::TVM);
  return TVMOpKernelPtr(OpKernelBuilder<TVMOpContext>()(field, op_name));
}

}
}
}

#endif // ONEFLOW_XRT_TVM_OPS_OP_KERNEL_H_
