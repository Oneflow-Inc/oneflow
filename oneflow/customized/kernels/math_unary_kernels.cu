#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include <math.h>

namespace oneflow {

namespace user_op {

#ifdef WITH_CUDA

template<typename T>
__device__ T AbsCalInDiff4Gpu(T x, T dy) {
  return x < 0 ? -dy : dy;
}

__device__ float AcosCalInDiff4GpuFloat(float x, float dy) { return dy * (-rsqrtf(1.0 - x * x)); }

__device__ float AcoshCalInDiff4GpuFloat(float x, float dy) { return dy * (rsqrtf(x * x - 1.0)); }

__device__ float AsinCalInDiff4GpuFloat(float x, float dy) { return dy * (rsqrtf(1.0 - x * x)); }

__device__ float AsinhCalInDiff4GpuFloat(float x, float dy) { return dy * (rsqrtf(1.0 + x * x)); }

__device__ float AtanCalInDiff4GpuFloat(float x, float dy) { return dy * (1.0 / (1.0 + x * x)); }

__device__ float AtanhCalInDiff4GpuFloat(float x, float dy) { return dy * (1.0 / (1.0 - x * x)); }

__device__ float CeilCalInDiff4GpuFloat(float x, float dy) { return 0.0; }

__device__ float CosCalInDiff4GpuFloat(float x, float dy) { return dy * (-sinf(x)); }

__device__ float CoshCalInDiff4GpuFloat(float x, float dy) {
  return dy * (expf(x) + expf(-x)) / 2.0;
}

__device__ float ErfCalInDiff4GpuFloat(float x, float dy) {
  return dy * 2.0 * rsqrtf(M_PI) * expf(-x * x);
}

__device__ float ErfcCalInDiff4GpuFloat(float x, float dy) {
  return dy * -2.0 * rsqrtf(M_PI) * expf(-x * x);
}

__device__ float ExpCalInDiff4GpuFloat(float x, float dy) { return dy * expf(x); }

__device__ float Expm1CalInDiff4GpuFloat(float x, float dy) { return dy * expf(x); }

__device__ float FloorCalInDiff4GpuFloat(float x, float dy) { return 0.0; }

__device__ float LgammaCalInDiff4GpuFloat(float x, float dy) {
  // TODO(chengcheng): return: dy * digamma(x)
  assert(false);
  return 0.0;
}

__device__ float LogCalInDiff4GpuFloat(float x, float dy) { return dy * (1.0 / x); }

__device__ float Log1pCalInDiff4GpuFloat(float x, float dy) { return dy * (1.0 / (x + 1.0)); }

__device__ float LogSigmoid4GpuFloat(float x) { return logf(1.0 / (1.0 + expf(-x))); }

__device__ float LogSigmoidCalInDiff4GpuFloat(float x, float dy) {
  return dy * (1.0 / (expf(x) + 1.0));
}

__device__ float Negative4GpuFloat(float x) { return -x; }

__device__ float NegativeCalInDiff4GpuFloat(float x, float dy) { return -dy; }

__device__ float Reciprocal4GpuFloat(float x) { return 1.0 / x; }

__device__ float ReciprocalCalInDiff4GpuFloat(float x, float dy) { return dy * (-1.0 / (x * x)); }

__device__ float ReciprocalNoNan4GpuFloat(float x) {
  if (fabsf(x) <= 0.0) { return 0.0; }
  return 1.0 / x;
}

__device__ float ReciprocalNoNanCalInDiff4GpuFloat(float x, float dy) {
  if (fabsf(x) <= 0.0) { return 0.0; }
  return dy * (-1.0 / (x * x));
}
__device__ float RintCalInDiff4GpuFloat(float x, float dy) { return 0.0; }

__device__ float RoundCalInDiff4GpuFloat(float x, float dy) { return 0.0; }

__device__ float RsqrtCalInDiff4GpuFloat(float x, float dy) {
  return dy * (-1.0 / (2.0 * sqrtf(x * x * x)));
}

#define MATH_UNARY_GPU(func_name, fw_func, bw_func, dtype)                                  \
  __global__ void func_name##ForwardGpu(const int n, const dtype* x, dtype* y) {            \
    CUDA_1D_KERNEL_LOOP(i, n) { y[i] = fw_func(x[i]); }                                     \
  }                                                                                         \
  void func_name##Forward(DeviceCtx* ctx, const Tensor* tensor_x, Tensor* tensor_y) {       \
    const dtype* x = tensor_x->dptr<dtype>();                                               \
    dtype* y = tensor_y->mut_dptr<dtype>();                                                 \
    int64_t n = tensor_x->shape().elem_cnt();                                               \
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);                                                  \
    func_name##ForwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,            \
                            ctx->cuda_stream()>>>(n, x, y);                                 \
  }                                                                                         \
  __global__ void func_name##BackwardGpu(const int n, const dtype* x, const dtype* dy,      \
                                         dtype* dx) {                                       \
    CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = bw_func(x[i], dy[i]); }                             \
  }                                                                                         \
  void func_name##Backward(DeviceCtx* ctx, const Tensor* tensor_x, const Tensor* tensor_dy, \
                           Tensor* tensor_dx) {                                             \
    const dtype* x = tensor_x->dptr<dtype>();                                               \
    const dtype* dy = tensor_dy->dptr<dtype>();                                             \
    dtype* dx = tensor_dx->mut_dptr<dtype>();                                               \
    int64_t n = tensor_x->shape().elem_cnt();                                               \
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);                                                  \
    func_name##BackwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,           \
                             ctx->cuda_stream()>>>(n, x, dy, dx);                           \
  }

#define MATH_UNARY_GPU_FLOAT_SEQ                           \
  OF_PP_MAKE_TUPLE_SEQ("Abs", Abs)                         \
  OF_PP_MAKE_TUPLE_SEQ("Acos", Acos)                       \
  OF_PP_MAKE_TUPLE_SEQ("Acosh", Acosh)                     \
  OF_PP_MAKE_TUPLE_SEQ("Asin", Asin)                       \
  OF_PP_MAKE_TUPLE_SEQ("Asinh", Asinh)                     \
  OF_PP_MAKE_TUPLE_SEQ("Atan", Atan)                       \
  OF_PP_MAKE_TUPLE_SEQ("Atanh", Atanh)                     \
  OF_PP_MAKE_TUPLE_SEQ("Ceil", Ceil)                       \
  OF_PP_MAKE_TUPLE_SEQ("Cos", Cos)                         \
  OF_PP_MAKE_TUPLE_SEQ("Cosh", Cosh)                       \
  OF_PP_MAKE_TUPLE_SEQ("Erf", Erf)                         \
  OF_PP_MAKE_TUPLE_SEQ("Erfc", Erfc)                       \
  OF_PP_MAKE_TUPLE_SEQ("Exp", Exp)                         \
  OF_PP_MAKE_TUPLE_SEQ("Expm1", Expm1)                     \
  OF_PP_MAKE_TUPLE_SEQ("Floor", Floor)                     \
  OF_PP_MAKE_TUPLE_SEQ("Lgamma", Lgamma)                   \
  OF_PP_MAKE_TUPLE_SEQ("Log", Log)                         \
  OF_PP_MAKE_TUPLE_SEQ("Log1p", Log1p)                     \
  OF_PP_MAKE_TUPLE_SEQ("LogSigmoid", LogSigmoid)           \
  OF_PP_MAKE_TUPLE_SEQ("Negative", Negative)               \
  OF_PP_MAKE_TUPLE_SEQ("Reciprocal", Reciprocal)           \
  OF_PP_MAKE_TUPLE_SEQ("ReciprocalNoNan", ReciprocalNoNan) \
  OF_PP_MAKE_TUPLE_SEQ("Rint", Rint)                       \
  OF_PP_MAKE_TUPLE_SEQ("Round", Round)                     \
  OF_PP_MAKE_TUPLE_SEQ("Rsqrt", Rsqrt)

MATH_UNARY_GPU(Abs, fabsf, AbsCalInDiff4Gpu<float>, float);
MATH_UNARY_GPU(Acos, acosf, AcosCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Acosh, acoshf, AcoshCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Asin, asinf, AsinCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Asinh, asinhf, AsinhCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Atan, atanf, AtanCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Atanh, atanhf, AtanhCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Ceil, ceilf, CeilCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Cos, cosf, CosCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Cosh, coshf, CoshCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Erf, erff, ErfCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Erfc, erfcf, ErfcCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Exp, expf, ExpCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Expm1, expm1f, Expm1CalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Floor, floorf, FloorCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Lgamma, lgammaf, LgammaCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Log, logf, LogCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Log1p, log1pf, Log1pCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(LogSigmoid, LogSigmoid4GpuFloat, LogSigmoidCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Negative, Negative4GpuFloat, NegativeCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Reciprocal, Reciprocal4GpuFloat, ReciprocalCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(ReciprocalNoNan, ReciprocalNoNan4GpuFloat, ReciprocalNoNanCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Rint, rintf, RintCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Round, roundf, RoundCalInDiff4GpuFloat, float);
MATH_UNARY_GPU(Rsqrt, rsqrtf, RsqrtCalInDiff4GpuFloat, float);

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

#define MATH_UNARY_FORWARD(unary_math_type_str, func_name_prefix)     \
  if (unary_math_type == unary_math_type_str) {                       \
    func_name_prefix##Forward(ctx->device_ctx(), tensor_x, tensor_y); \
  }

    OF_PP_FOR_EACH_TUPLE(MATH_UNARY_FORWARD, MATH_UNARY_GPU_FLOAT_SEQ);
#undef MATH_UNARY_FORWARD
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
    const Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    std::string unary_math_type = ctx->GetAttr<std::string>("unary_math_type");

#define MATH_UNARY_BACKWARD(unary_math_type_str, func_name_prefix)                 \
  if (unary_math_type == unary_math_type_str) {                                    \
    func_name_prefix##Backward(ctx->device_ctx(), tensor_x, tensor_dy, tensor_dx); \
  }

    OF_PP_FOR_EACH_TUPLE(MATH_UNARY_BACKWARD, MATH_UNARY_GPU_FLOAT_SEQ);
#undef MATH_UNARY_BACKWARD
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
