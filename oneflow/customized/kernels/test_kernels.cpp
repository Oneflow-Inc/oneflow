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
    .SetInferTmpSizeFn([](user_op::InferContext*) { return 10; })
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));
      return Maybe<void>::Ok();
    });

REGISTER_USER_KERNEL("ccrelu_grad")
    .SetCreateFn([](const user_op::KernelInitContext& ctx) { return new ReluGradKernel(ctx); })
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) { return true; })
    .SetInferTmpSizeFn([](user_op::InferContext*) { return 10; });

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

class CopyIn2OutKernel final : public user_op::OpKernel {
 public:
  CopyIn2OutKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  CopyIn2OutKernel() = default;
  ~CopyIn2OutKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), out_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                             in_blob->shape().elem_cnt() * sizeof(float));
  }
};

REGISTER_USER_KERNEL("TestReshape4KeepHeaderOnly")
    .SetCreateFn([](const user_op::KernelInitContext& ctx) { return new CopyIn2OutKernel(ctx); })
    .SetIsMatchedPred([](const user_op::KernelRegContext&) { return true; });

REGISTER_USER_KERNEL("TestReshapeLike4KeepHeaderOnly")
    .SetCreateFn([](const user_op::KernelInitContext& ctx) { return new CopyIn2OutKernel(ctx); })
    .SetIsMatchedPred([](const user_op::KernelRegContext&) { return true; });

class TestSourceKernel final : public user_op::OpKernel {
 public:
  TestSourceKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  TestSourceKernel() = default;
  ~TestSourceKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    for (int i = 0; i < 5; ++i) { *(out_blob->mut_dptr<float>() + i) = static_cast<float>(i); }
  }
};

REGISTER_USER_KERNEL("TestSource")
    .SetCreateFn([](const user_op::KernelInitContext& ctx) { return new TestSourceKernel(ctx); })
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
      if (ctx.device_type() == DeviceType::kCPU && out_tensor->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
    })
    .SetInferTmpSizeFn([](user_op::InferContext*) { return 0; });

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
    .SetCreateFn([](const user_op::KernelInitContext& ctx) {
      return new TestMultiOutputOrderKernel(ctx);
    })
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* in_tensor = ctx.TensorDesc4ArgNameAndIndex("in", 0);
      if (ctx.device_type() == DeviceType::kGPU && in_tensor->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
    });

class TestSourceMultiGpuFixedOutNumKernel final : public user_op::OpKernel {
 public:
  TestSourceMultiGpuFixedOutNumKernel(const user_op::KernelInitContext& ctx)
      : user_op::OpKernel(ctx) {}
  TestSourceMultiGpuFixedOutNumKernel() = default;
  ~TestSourceMultiGpuFixedOutNumKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    for (int i = 0; i < out_blob->shape().elem_cnt(); ++i) {
      *(out_blob->mut_dptr<float>() + i) = static_cast<float>(i);
    }
  }
};

REGISTER_USER_KERNEL("TestSourceMultiGpuFixedOutNum")
    .SetCreateFn([](const user_op::KernelInitContext& ctx) {
      return new TestSourceMultiGpuFixedOutNumKernel(ctx);
    })
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
      if (ctx.device_type() == DeviceType::kCPU && out_tensor->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
    });

class TestMultiInputFwKernel final : public user_op::OpKernel {
 public:
  TestMultiInputFwKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  TestMultiInputFwKernel() = default;
  ~TestMultiInputFwKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x1_blob = ctx->Tensor4ArgNameAndIndex("x1", 0);
    user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), y_blob->mut_dptr<char>(), x1_blob->dptr<char>(),
                             x1_blob->shape().elem_cnt() * sizeof(float));
  }
};

REGISTER_USER_KERNEL("TestMultiInput")
    .SetCreateFn([](const user_op::KernelInitContext& ctx) {
      return new TestMultiInputFwKernel(ctx);
    })
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* x1_tensor = ctx.TensorDesc4ArgNameAndIndex("x1", 0);
      if (ctx.device_type() == DeviceType::kGPU && x1_tensor->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
    });

class TestMultiInputBwKernel final : public user_op::OpKernel {
 public:
  TestMultiInputBwKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  TestMultiInputBwKernel() = default;
  ~TestMultiInputBwKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    user_op::Tensor* x1_diff_blob = ctx->Tensor4ArgNameAndIndex("x1_diff", 0);
    user_op::Tensor* x2_diff_blob = ctx->Tensor4ArgNameAndIndex("x2_diff", 0);
    NewKernelUtil<DeviceType::kGPU>::Fill(ctx->device_ctx(), x1_diff_blob->shape().elem_cnt(), 1.0,
                                          x1_diff_blob->mut_dptr<float>());
    NewKernelUtil<DeviceType::kGPU>::Fill(ctx->device_ctx(), x2_diff_blob->shape().elem_cnt(), 2.0,
                                          x2_diff_blob->mut_dptr<float>());
  }
};

REGISTER_USER_KERNEL("TestMultiInputGrad")
    .SetCreateFn([](const user_op::KernelInitContext& ctx) {
      return new TestMultiInputBwKernel(ctx);
    })
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* x1_tensor = ctx.TensorDesc4ArgNameAndIndex("x1", 0);
      if (ctx.device_type() == DeviceType::kGPU && x1_tensor->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
    });

}  // namespace oneflow
