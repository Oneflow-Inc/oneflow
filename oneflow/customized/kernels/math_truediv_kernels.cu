#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include <math.h>

namespace oneflow {

namespace user_op {

#ifdef WITH_CUDA

__device__ float TruedivCalXDiff4GpuFloat(float x, float y, float dz) { return dz * fdivide(1, y); }

__device__ float TruedivCalYDiff4GpuFloat(float x, float y, float dz) {
  return dz * (-1) * fdivide(x, powf(y, 2));
}

#define MATH_TRUEDIV_GPU(func_name, fw_func, bw_func_cal_x_diff, bw_func_cal_y_diff, dtype)      \
  __global__ void func_name##ForwardGpu(const int n, const dtype* x, const dtype* y, dtype* z) { \
    CUDA_1D_KERNEL_LOOP(i, n) { z[i] = fw_func(x[i], y[i]); }                                    \
  }                                                                                              \
  void func_name##Forward(DeviceCtx* ctx, const Tensor* tensor_x, const Tensor* tensor_y,        \
                          Tensor* tensor_z) {                                                    \
    const dtype* x = tensor_x->dptr<dtype>();                                                    \
    const dtype* y = tensor_y->dptr<dtype>();                                                    \
    dtype* z = tensor_z->mut_dptr<dtype>();                                                      \
    int64_t n = tensor_x->shape().elem_cnt();                                                    \
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);                                                       \
    func_name##ForwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,                 \
                            ctx->cuda_stream()>>>(n, x, y, z);                                   \
  }                                                                                              \
  __global__ void func_name##XBackwardGpu(const int n, const dtype* x, const dtype* y,           \
                                          const dtype* dz, dtype* dx) {                          \
    CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = bw_func_cal_x_diff(x[i], y[i], dz[i]); }                 \
  }                                                                                              \
  void func_name##XBackward(DeviceCtx* ctx, const Tensor* tensor_x, const Tensor* tensor_y,      \
                            const Tensor* tensor_dz, Tensor* tensor_dx) {                        \
    const dtype* x = tensor_x->dptr<dtype>();                                                    \
    const dtype* y = tensor_y->dptr<dtype>();                                                    \
    const dtype* dz = tensor_dz->dptr<dtype>();                                                  \
    dtype* dx = tensor_dx->mut_dptr<dtype>();                                                    \
    int64_t n = tensor_x->shape().elem_cnt();                                                    \
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);                                                       \
    func_name##XBackwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,               \
                              ctx->cuda_stream()>>>(n, x, y, dz, dx);                            \
  }                                                                                              \
  __global__ void func_name##YBackwardGpu(const int n, const dtype* x, const dtype* y,           \
                                          const dtype* dz, dtype* dy) {                          \
    CUDA_1D_KERNEL_LOOP(i, n) { dy[i] = bw_func_cal_y_diff(x[i], y[i], dz[i]); }                 \
  }                                                                                              \
  void func_name##YBackward(DeviceCtx* ctx, const Tensor* tensor_x, const Tensor* tensor_y,      \
                            const Tensor* tensor_dz, Tensor* tensor_dy) {                        \
    const dtype* x = tensor_x->dptr<dtype>();                                                    \
    const dtype* y = tensor_y->dptr<dtype>();                                                    \
    const dtype* dz = tensor_dz->dptr<dtype>();                                                  \
    dtype* dy = tensor_dy->mut_dptr<dtype>();                                                    \
    int64_t n = tensor_x->shape().elem_cnt();                                                    \
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);                                                       \
    func_name##YBackwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,               \
                              ctx->cuda_stream()>>>(n, x, y, dz, dy);                            \
  }

#define MATH_TRUEDIV_GPU_FLOAT_SEQ OF_PP_MAKE_TUPLE_SEQ(Truediv)

MATH_TRUEDIV_GPU(Truediv, fdivide, TruedivCalXDiff4GpuFloat, TruedivCalYDiff4GpuFloat, float);

class MathTruedivGpuFloatKernel final : public OpKernel {
 public:
  MathTruedivGpuFloatKernel(const KernelInitContext& ctx) : OpKernel(ctx) {}
  MathTruedivGpuFloatKernel() = default;
  ~MathTruedivGpuFloatKernel() = default;

 private:
  void Compute(KernelContext* ctx) override {
    const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);

#define MATH_TRUEDIV_FORWARD(func_name_prefix) \
  func_name_prefix##Forward(ctx->device_ctx(), tensor_x, tensor_y, tensor_z);

    OF_PP_FOR_EACH_TUPLE(MATH_TRUEDIV_FORWARD, MATH_TRUEDIV_GPU_FLOAT_SEQ);
#undef MATH_TRUEDIV_FORWARD
  }
};

REGISTER_USER_KERNEL("truediv")
    .SetCreateFn([](const KernelInitContext& ctx) { return new MathTruedivGpuFloatKernel(ctx); })
    .SetIsMatchedPred([](const KernelRegContext& ctx) {
      const user_op::TensorDesc* x_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);
      const user_op::TensorDesc* y_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);
      if (ctx.device() == DeviceType::kGPU && x_tensor_desc->data_type() == DataType::kFloat
          && y_tensor_desc->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
    });

class MathTruedivXGradGpuFloatKernel final : public OpKernel {
 public:
  MathTruedivXGradGpuFloatKernel(const KernelInitContext& ctx) : OpKernel(ctx) {}
  MathTruedivXGradGpuFloatKernel() = default;
  ~MathTruedivXGradGpuFloatKernel() = default;

 private:
  void Compute(KernelContext* ctx) override {
    const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

#define MATH_TRUEDIV_BACKWARD(func_name_prefix) \
  func_name_prefix##XBackward(ctx->device_ctx(), tensor_x, tensor_y, tensor_dz, tensor_dx);

    OF_PP_FOR_EACH_TUPLE(MATH_TRUEDIV_BACKWARD, MATH_TRUEDIV_GPU_FLOAT_SEQ);
#undef MATH_TRUEDIV_FORWARD
  }
};

class MathTruedivYGradGpuFloatKernel final : public OpKernel {
 public:
  MathTruedivYGradGpuFloatKernel(const KernelInitContext& ctx) : OpKernel(ctx) {}
  MathTruedivYGradGpuFloatKernel() = default;
  ~MathTruedivYGradGpuFloatKernel() = default;

 private:
  void Compute(KernelContext* ctx) override {
    const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);

#define MATH_TRUEDIV_BACKWARD(func_name_prefix) \
  func_name_prefix##YBackward(ctx->device_ctx(), tensor_x, tensor_y, tensor_dz, tensor_dy);

    OF_PP_FOR_EACH_TUPLE(MATH_TRUEDIV_BACKWARD, MATH_TRUEDIV_GPU_FLOAT_SEQ);
#undef MATH_TRUEDIV_FORWARD
  }
};

REGISTER_USER_KERNEL("truediv_x_grad")
    .SetCreateFn([](const KernelInitContext& ctx) {
      return new MathTruedivXGradGpuFloatKernel(ctx);
    })
    .SetIsMatchedPred([](const KernelRegContext& ctx) {
      const user_op::TensorDesc* x_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);
      if (ctx.device() == DeviceType::kGPU && x_tensor_desc->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
    });

REGISTER_USER_KERNEL("truediv_y_grad")
    .SetCreateFn([](const KernelInitContext& ctx) {
      return new MathTruedivYGradGpuFloatKernel(ctx);
    })
    .SetIsMatchedPred([](const KernelRegContext& ctx) {
      const user_op::TensorDesc* y_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);
      if (ctx.device() == DeviceType::kGPU && y_tensor_desc->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
    });

#endif  // WITH_CUDA

}  // namespace user_op

}  // namespace oneflow
