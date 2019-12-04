#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

class ReluKernel final : public oneflow::user_op::OpKernel {
 public:
  ReluKernel(const oneflow::user_op::KernelInitContext& ctx) : oneflow::user_op::OpKernel(ctx) {}
  ReluKernel() = default;
  ~ReluKernel() = default;

 private:
  void Compute(oneflow::user_op::KernelContext* ctx) override {
    const oneflow::user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    oneflow::user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    oneflow::NewKernelUtil<oneflow::DeviceType::kGPU>::Relu(
        ctx->device_ctx(), in_blob->shape().elem_cnt(), in_blob->dptr<float>(),
        out_blob->mut_dptr<float>());
  }
};

class ReluGradKernel final : public oneflow::user_op::OpKernel {
 public:
  ReluGradKernel(const oneflow::user_op::KernelInitContext& ctx)
      : oneflow::user_op::OpKernel(ctx) {}
  ReluGradKernel() = default;
  ~ReluGradKernel() = default;

 private:
  void Compute(oneflow::user_op::KernelContext* ctx) override {
    const oneflow::user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const oneflow::user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    oneflow::user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    oneflow::NewKernelUtil<oneflow::DeviceType::kGPU>::ReluBackward(
        ctx->device_ctx(), dx_blob->shape().elem_cnt(), y_blob->dptr<float>(),
        y_blob->dptr<float>(), dy_blob->dptr<float>(), dx_blob->mut_dptr<float>());
  }
};

REGISTER_USER_KERNEL("ccrelu")
    .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) { return new ReluKernel(ctx); })
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) { return true; })
    .SetInferTmpSizeFn([](const oneflow::user_op::InferContext&) { return 10; });

REGISTER_USER_KERNEL("ccrelu_grad")
    .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {
      return new ReluGradKernel(ctx);
    })
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) { return true; })
    .SetInferTmpSizeFn([](const oneflow::user_op::InferContext&) { return 10; });

class TestReshapeKernel final : public oneflow::user_op::OpKernel {
 public:
  TestReshapeKernel(const oneflow::user_op::KernelInitContext& ctx)
      : oneflow::user_op::OpKernel(ctx) {}
  TestReshapeKernel() = default;
  ~TestReshapeKernel() = default;

 private:
  void Compute(oneflow::user_op::KernelContext* ctx) override {
    // const oneflow::user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    // oneflow::user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    LOG(WARNING) << "Run TestReshape Kernel";
  }
};

REGISTER_USER_KERNEL("TestReshape")
    .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {
      return new TestReshapeKernel(ctx);
    })
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext&) { return true; });
