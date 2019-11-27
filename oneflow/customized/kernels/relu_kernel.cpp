#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

class ReluKernel final : public oneflow::user_op::OpKernel {
 public:
  ReluKernel() = default;
  ~ReluKernel() = default;

 private:
  void Compute(const oneflow::user_op::KernelContext& ctx) const override {
    LOG(WARNING) << "Run Relu Kernel";
  }
};

REGISTER_USER_KERNEL("ccrelu")
    .SetCreateFn([](const oneflow::user_op::KernelInitContext&) { return new ReluKernel(); })
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) { return true; })
    .SetInferTmpSizeFn([]() { return 10; });
