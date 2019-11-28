#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

class ReluKernel final : public oneflow::user_op::OpKernel {
 public:
  ReluKernel(const oneflow::user_op::KernelInitContext& ctx) : oneflow::user_op::OpKernel(ctx) {}
  ReluKernel() = default;
  ~ReluKernel() = default;

 private:
  void Compute(oneflow::user_op::KernelContext* ctx) override {
    const oneflow::user_op::Blob* in_blob = ctx->Blob4ArgNameAndIndex("in", 0);
    oneflow::user_op::Blob* out_blob = ctx->Blob4ArgNameAndIndex("out", 0);
    oneflow::NewKernelUtil<oneflow::DeviceType::kGPU>::Relu(
        ctx->device_ctx(), in_blob->shape().elem_cnt(), in_blob->dptr<float>(),
        out_blob->mut_dptr<float>());
  }
};

REGISTER_USER_KERNEL("ccrelu")
    .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) { return new ReluKernel(ctx); })
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) { return true; })
    .SetInferTmpSizeFn([]() { return 10; });
