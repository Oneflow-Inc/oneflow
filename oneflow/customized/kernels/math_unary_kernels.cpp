#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace user_op {

#ifdef WITH_CUDA

class MathUnaryGpuFloatKernel final : public OpKernel {
 public:
  MathUnaryGpuFloatKernel(const KernelInitContext& ctx) : OpKernel(ctx) {}
  MathUnaryGpuFloatKernel() = default;
  ~MathUnaryGpuFloatKernel() = default;

 private:
  void Compute(KernelContext* ctx) override {
    const Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    TODO();
    // NewKernelUtil<DeviceType::kGPU>::Relu(
    //    ctx->device_ctx(), in_blob->shape().elem_cnt(), in_blob->dptr<float>(),
    //    out_blob->mut_dptr<float>());
  }
};
/*
Maybe<void> Foo(const InferContext&, AddInplaceArgPair AddInplaceArgPairFn) {
  JUST(AddInplaceArgPairFn("y", 0, "x", 0, true));
  return Maybe<void>::Ok();
}
*/
REGISTER_USER_KERNEL("unary")
    .SetCreateFn([](const KernelInitContext& ctx) { return new MathUnaryGpuFloatKernel(ctx); })
    .SetIsMatchedPred([](const KernelRegContext& ctx) {
      if (ctx.device() == DeviceType::kGPU && ctx.data_type() == DataType::kFloat) { return true; }
      return false;
    })
    .SetInplaceProposalFn([](const InferContext&,
                             AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {
      // return AddInplaceArgPairFn("y", 0, "x", 0, true);
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));
      // AddInplaceArgPairFn("y", 0, "x", 0, true);
      return Maybe<void>::Ok();
    });
#endif  // WITH_CUDA

/*
class ReluGradKernel final : public OpKernel {
 public:
  ReluGradKernel(const KernelInitContext& ctx)
      : OpKernel(ctx) {}
  ReluGradKernel() = default;
  ~ReluGradKernel() = default;

 private:
  void Compute(KernelContext* ctx) override {
    const Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    NewKernelUtil<DeviceType::kGPU>::ReluBackward(
        ctx->device_ctx(), dx_blob->shape().elem_cnt(), y_blob->dptr<float>(),
        y_blob->dptr<float>(), dy_blob->dptr<float>(), dx_blob->mut_dptr<float>());
  }
};


REGISTER_USER_KERNEL("ccrelu_grad")
    .SetCreateFn([](const KernelInitContext& ctx) {
      return new ReluGradKernel(ctx);
    })
    .SetIsMatchedPred([](const KernelRegContext& ctx) { return true; })
    .SetInferTmpSizeFn([](const InferContext&) { return 10; });

class TestReshapeKernel final : public OpKernel {
 public:
  TestReshapeKernel(const KernelInitContext& ctx)
      : OpKernel(ctx) {}
  TestReshapeKernel() = default;
  ~TestReshapeKernel() = default;

 private:
  void Compute(KernelContext* ctx) override {
    const Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), out_blob->mut_dptr<char>(),
                                               in_blob->dptr<char>(),
                                               in_blob->shape().elem_cnt() * sizeof(float));
  }
};

REGISTER_USER_KERNEL("TestReshape")
    .SetCreateFn([](const KernelInitContext& ctx) {
      return new TestReshapeKernel(ctx);
    })
    .SetIsMatchedPred([](const KernelRegContext&) { return true; });
*/
}  // namespace user_op

}  // namespace oneflow
