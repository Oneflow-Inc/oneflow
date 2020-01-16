#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace user_op {

#ifdef WITH_CUDA

template<typename T>
__device__ T AbsCalInDiff4Gpu(T x, T y, T dy) {
  return x < 0 ? -dy : dy;
}

__device__ float AcosCalInDiff4GpuFloat(float x, float y, float dy) {
  return dy * (-rsqrtf(1 - x * x));
}

#define MATH_UNARY_GPU(func_name, fw_func, bw_func, dtype)                                 \
  __global__ void func_name##ForwardGpu(const int n, const dtype* x, dtype* y) {           \
    CUDA_1D_KERNEL_LOOP(i, n) { y[i] = fw_func(x[i]); }                                    \
  }                                                                                        \
  void func_name##Forward(DeviceCtx* ctx, const Tensor* tensor_x, Tensor* tensor_y) {      \
    const dtype* x = tensor_x->dptr<dtype>();                                              \
    dtype* y = tensor_y->mut_dptr<dtype>();                                                \
    int64_t n = tensor_x->shape().elem_cnt();                                              \
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);                                                 \
    func_name##ForwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,           \
                            ctx->cuda_stream()>>>(n, x, y);                                \
  }                                                                                        \
  __global__ void func_name##BackwardGpu(const int n, const dtype* x, const dtype* y,      \
                                         const dtype* dy, dtype* dx) {                     \
    CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = bw_func(x[i], y[i], dy[i]); }                      \
  }                                                                                        \
  void func_name##Backward(DeviceCtx* ctx, const Tensor* tensor_x, const Tensor* tensor_y, \
                           const Tensor* tensor_dy, Tensor* tensor_dx) {                   \
    const dtype* x = tensor_x->dptr<dtype>();                                              \
    const dtype* y = tensor_y->dptr<dtype>();                                              \
    const dtype* dy = tensor_dy->dptr<dtype>();                                            \
    dtype* dx = tensor_dx->mut_dptr<dtype>();                                              \
    int64_t n = tensor_x->shape().elem_cnt();                                              \
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);                                                 \
    func_name##BackwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,          \
                             ctx->cuda_stream()>>>(n, x, y, dy, dx);                       \
  }

MATH_UNARY_GPU(Abs, fabsf, AbsCalInDiff4Gpu<float>, float);
MATH_UNARY_GPU(Acos, acosf, AcosCalInDiff4GpuFloat, float);
/*
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
*/
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
    if (unary_math_type == "Abs") { AbsForward(ctx->device_ctx(), tensor_x, tensor_y); }
    if (unary_math_type == "Acos") { AcosForward(ctx->device_ctx(), tensor_x, tensor_y); }
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

class MathUnaryGradGpuFloatKernel final : public OpKernel {
 public:
  MathUnaryGradGpuFloatKernel(const KernelInitContext& ctx) : OpKernel(ctx) {}
  MathUnaryGradGpuFloatKernel() = default;
  ~MathUnaryGradGpuFloatKernel() = default;

 private:
  void Compute(KernelContext* ctx) override {
    const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    std::string unary_math_type = ctx->GetAttr<std::string>("unary_math_type");
    if (unary_math_type == "Abs") {
      AbsBackward(ctx->device_ctx(), tensor_x, tensor_y, tensor_dy, tensor_dx);
    }
    if (unary_math_type == "Acos") {
      AcosBackward(ctx->device_ctx(), tensor_x, tensor_y, tensor_dy, tensor_dx);
    }
  }
};

REGISTER_USER_KERNEL("unary_grad")
    .SetCreateFn([](const KernelInitContext& ctx) { return new MathUnaryGradGpuFloatKernel(ctx); })
    .SetIsMatchedPred([](const KernelRegContext& ctx) {
      if (ctx.device() == DeviceType::kGPU && ctx.data_type() == DataType::kFloat) { return true; }
      return false;
    });

#endif  // WITH_CUDA

}  // namespace user_op

}  // namespace oneflow
