#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

class ReluKernel final : public user_op::OpKernel {
 public:
  ReluKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  ReluKernel() = default;
  ~ReluKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    CHECK_NOTNULL(tmp);
    NewKernelUtil<DeviceType::kGPU>::Relu(ctx->device_ctx(), in_blob->shape().elem_cnt(),
                                          in_blob->dptr<float>(), out_blob->mut_dptr<float>());
  }
};

class ReluGradKernel final : public user_op::OpKernel {
 public:
  ReluGradKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  ReluGradKernel() = default;
  ~ReluGradKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    NewKernelUtil<DeviceType::kGPU>::ReluBackward(
        ctx->device_ctx(), dx_blob->shape().elem_cnt(), y_blob->dptr<float>(),
        y_blob->dptr<float>(), dy_blob->dptr<float>(), dx_blob->mut_dptr<float>());
  }
};

REGISTER_USER_KERNEL("ccrelu")
    .SetCreateFn([](const user_op::KernelInitContext& ctx) { return new ReluKernel(ctx); })
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) { return true; })
    .SetInferTmpSizeFn([](const user_op::InferContext&) { return 10; })
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));
      return Maybe<void>::Ok();
    });

REGISTER_USER_KERNEL("ccrelu_grad")
    .SetCreateFn([](const user_op::KernelInitContext& ctx) { return new ReluGradKernel(ctx); })
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) { return true; })
    .SetInferTmpSizeFn([](const user_op::InferContext&) { return 10; });

class TestReshapeKernel final : public user_op::OpKernel {
 public:
  TestReshapeKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  TestReshapeKernel() = default;
  ~TestReshapeKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), out_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                             in_blob->shape().elem_cnt() * sizeof(float));
  }
};

REGISTER_USER_KERNEL("TestReshape")
    .SetCreateFn([](const user_op::KernelInitContext& ctx) { return new TestReshapeKernel(ctx); })
    .SetIsMatchedPred([](const user_op::KernelRegContext&) { return true; });

class TestSourceKernel final : public oneflow::user_op::OpKernel {
 public:
  TestSourceKernel(const oneflow::user_op::KernelInitContext& ctx)
      : oneflow::user_op::OpKernel(ctx) {}
  TestSourceKernel() = default;
  ~TestSourceKernel() = default;

 private:
  void Compute(oneflow::user_op::KernelContext* ctx) override {
    oneflow::user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    for (int i = 0; i < 5; ++i) { *(out_blob->mut_dptr<float>() + i) = static_cast<float>(i); }
  }
};

REGISTER_USER_KERNEL("TestSource")
    .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {
      return new TestSourceKernel(ctx);
    })
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {
      if (ctx.device() == DeviceType::kCPU && ctx.data_type() == DataType::kFloat) { return true; }
      return false;
    })
    .SetInferTmpSizeFn([](const oneflow::user_op::InferContext&) { return 0; });

class TestMultiOutputOrderKernel final : public user_op::OpKernel {
 public:
  TestMultiOutputOrderKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  TestMultiOutputOrderKernel() = default;
  ~TestMultiOutputOrderKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out1_blob = ctx->Tensor4ArgNameAndIndex("out1", 0);
    user_op::Tensor* out2_blob = ctx->Tensor4ArgNameAndIndex("out2", 0);
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), out1_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                             in_blob->shape().elem_cnt() * sizeof(float));
    NewKernelUtil<DeviceType::kGPU>::Fill(ctx->device_ctx(), out2_blob->shape().elem_cnt(), 0.0,
                                          out2_blob->mut_dptr<float>());
  }
};

REGISTER_USER_KERNEL("TestMultiOutputOrder")
    .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {
      return new TestMultiOutputOrderKernel(ctx);
    })
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {
      if (ctx.device() == DeviceType::kGPU && ctx.data_type() == DataType::kFloat) { return true; }
      return false;
    });

}  // namespace oneflow
