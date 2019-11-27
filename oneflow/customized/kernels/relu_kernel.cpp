#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

class ReluKernel final : public oneflow::user_op::OpKernel {
 public:
  ReluKernel(const oneflow::user_op::KernelInitContext& ctx) : oneflow::user_op::OpKernel(ctx) {}
  ReluKernel() = default;
  ~ReluKernel() = default;

 private:
  void Compute(oneflow::user_op::KernelContext* ctx) override { LOG(WARNING) << "Run Relu Kernel"; }
};

REGISTER_USER_KERNEL("ccrelu")
    .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) { return new ReluKernel(ctx); })
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) { return true; })
    .SetInferTmpSizeFn([]() { return 10; });
