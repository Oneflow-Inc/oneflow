#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/math_unary_elementwise_func.h"

namespace oneflow {

namespace {

template<template<typename> class UnaryFunctor, typename T>
__global__ void MathUnaryElementwiseForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = UnaryFunctor<T>::Forward(x[i]); }
}

template<template<typename> class UnaryFunctor, typename T>
__global__ void MathUnaryElementwiseBackwardGpu(const int n, const T* x, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = UnaryFunctor<T>::Backward(x[i], dy[i]); }
}

template<template<typename> class UnaryFunctor, typename T>
class MathUnaryElementwiseGpuKernel final : public user_op::OpKernel {
 public:
  MathUnaryElementwiseGpuKernel() = default;
  ~MathUnaryElementwiseGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const T* x = tensor_x->dptr<T>();
    T* y = tensor_y->mut_dptr<T>();
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    MathUnaryElementwiseForwardGpu<UnaryFunctor, T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->device_ctx()->cuda_stream()>>>(
            n, x, y);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename> class UnaryFunctor, typename T>
class MathUnaryElementwiseGradGpuKernel final : public user_op::OpKernel {
 public:
  MathUnaryElementwiseGradGpuKernel() = default;
  ~MathUnaryElementwiseGradGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const T* x = tensor_x->dptr<T>();
    const T* dy = tensor_dy->dptr<T>();
    T* dx = tensor_dx->mut_dptr<T>();
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    MathUnaryElementwiseBackwardGpu<UnaryFunctor, T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->device_ctx()->cuda_stream()>>>(
            n, x, dy, dx);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("math_unary_elementwise")
    .SetCreateFn<MathUnaryElementwiseGpuKernel<AcosFunctor, float>>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* x_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);
      const user_op::TensorDesc* y_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);
      return ctx.device_type() == DeviceType::kGPU && x_tensor_desc->data_type() == DataType::kFloat
             && y_tensor_desc->data_type() == DataType::kFloat
             && ctx.GetAttr<std::string>("math_type") == "Acos";
    });
REGISTER_USER_KERNEL("math_unary_elementwise_grad")
    .SetCreateFn<MathUnaryElementwiseGradGpuKernel<AcosFunctor, float>>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* x_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);
      return ctx.device_type() == DeviceType::kGPU && x_tensor_desc->data_type() == DataType::kFloat
             && ctx.GetAttr<std::string>("math_type") == "Acos";
    });

REGISTER_USER_KERNEL("math_unary_elementwise")
    .SetCreateFn<MathUnaryElementwiseGpuKernel<AbsFunctor, float>>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* x_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);
      const user_op::TensorDesc* y_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);
      return ctx.device_type() == DeviceType::kGPU && x_tensor_desc->data_type() == DataType::kFloat
             && y_tensor_desc->data_type() == DataType::kFloat
             && ctx.GetAttr<std::string>("math_type") == "Abs";
    });
REGISTER_USER_KERNEL("math_unary_elementwise_grad")
    .SetCreateFn<MathUnaryElementwiseGradGpuKernel<AbsFunctor, float>>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* x_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);
      return ctx.device_type() == DeviceType::kGPU && x_tensor_desc->data_type() == DataType::kFloat
             && ctx.GetAttr<std::string>("math_type") == "Abs";
    });

}  // namespace

/*
template<typename T, OF_DEVICE_FUNC T (*unary_fw_func)(T x)>
__global__ void MathUnaryElementwiseForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = unary_fw_func(x[i]); }
}

template<typename T, OF_DEVICE_FUNC T (*unary_bw_func)(T x, T dy)>
__global__ void MathUnaryElementwiseBackwardGpu(const int n, const T* x, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = unary_bw_func(x[i], dy[i]); }
}

}  // namespace

template<typename T, OF_DEVICE_FUNC T (*unary_fw_func)(T x)>
class MathUnaryElementwiseGpuKernel final : public user_op::OpKernel {
 public:
  MathUnaryElementwiseGpuKernel() = default;
  ~MathUnaryElementwiseGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const T* x = tensor_x->dptr<T>();
    T* y = tensor_y->mut_dptr<T>();
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    MathUnaryElementwiseForwardGpu<T, unary_fw_func>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->device_ctx()->cuda_stream()>>>(
            n, x, y);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T, OF_DEVICE_FUNC T (*unary_bw_func)(T x, T dy)>
class MathUnaryElementwiseGradGpuKernel final : public user_op::OpKernel {
 public:
  MathUnaryElementwiseGradGpuKernel() = default;
  ~MathUnaryElementwiseGradGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const T* x = tensor_x->dptr<T>();
    const T* dy = tensor_dy->dptr<T>();
    T* dx = tensor_dx->mut_dptr<T>();
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    MathUnaryElementwiseBackwardGpu<T, unary_bw_func>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->device_ctx()->cuda_stream()>>>(
            n, x, dy, dx);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MATH_UNARY_ELEMETNWISE_KERNEL_ENTRY(math_type, fw_func, bw_func)          \
  REGISTER_USER_KERNEL("math_unary_elementwise")                                           \
      .SetCreateFn<MathUnaryElementwiseGpuKernel<float, &fw_func>>()                       \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                         \
        const user_op::TensorDesc* x_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0); \
        const user_op::TensorDesc* y_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0); \
        return ctx.device_type() == DeviceType::kGPU                                       \
               && x_tensor_desc->data_type() == DataType::kFloat                           \
               && y_tensor_desc->data_type() == DataType::kFloat                           \
               && ctx.GetAttr<std::string>("math_type") == math_type;                      \
      });                                                                                  \
  REGISTER_USER_KERNEL("math_unary_elementwise_grad")                                      \
      .SetCreateFn<MathUnaryElementwiseGradGpuKernel<float, &bw_func>>()                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                         \
        const user_op::TensorDesc* x_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0); \
        return ctx.device_type() == DeviceType::kGPU                                       \
               && x_tensor_desc->data_type() == DataType::kFloat                           \
               && ctx.GetAttr<std::string>("math_type") == math_type;                      \
      });

OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_ELEMETNWISE_KERNEL_ENTRY, MATH_UNARY_ELEMENTWISE_FUNC_SEQ)
*/
}  // namespace oneflow
