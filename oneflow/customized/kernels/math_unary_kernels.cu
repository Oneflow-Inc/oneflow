#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace user_op {

#ifdef WITH_CUDA

__global__ void AbsForwardGpu(const int n, const float* x, float* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = fabsf(x[i]); }
}

void Abs(DeviceCtx* ctx, const Tensor* tensor_x, Tensor* tensor_y) {
  const float* x = tensor_x->dptr<float>();
  float* y = tensor_y->mut_dptr<float>();
  int64_t n = tensor_x->shape().elem_cnt();
  CHECK_LE(n, GetMaxVal<int32_t>() / 2);
  AbsForwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x,
                                                                                             y);
}

class MathUnaryGpuFloatKernel final : public OpKernel {
 public:
  MathUnaryGpuFloatKernel(const KernelInitContext& ctx) : OpKernel(ctx) {}
  MathUnaryGpuFloatKernel() = default;
  ~MathUnaryGpuFloatKernel() = default;

 private:
  void Compute(KernelContext* ctx) override {
    const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    std::string unary_math_type = ctx->GetAttr<std::string>("unary_math_type");
    if (unary_math_type == "Abs") { Abs(ctx->device_ctx(), tensor_x, tensor_y); }
  }
};

REGISTER_USER_KERNEL("unary")
    .SetCreateFn([](const KernelInitContext& ctx) { return new MathUnaryGpuFloatKernel(ctx); })
    .SetIsMatchedPred([](const KernelRegContext& ctx) {
      if (ctx.device() == DeviceType::kGPU && ctx.data_type() == DataType::kFloat) { return true; }
      return false;
    });
/*
    .SetInplaceProposalFn([](const InferContext&,
                             AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));
      return Maybe<void>::Ok();
    });
*/
#endif  // WITH_CUDA

}  // namespace user_op

}  // namespace oneflow
