#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include <math.h>

namespace oneflow {

namespace user_op {

#ifdef WITH_CUDA

__device__ float PowCalXDiff4GpuFloat(float x, float y, float dz) {
  return dz * y * (powf(x, y - 1));
}

__device__ float PowCalYDiff4GpuFloat(float x, float y, float dz) {
  if (x > 0) {
    return dz * logf(x) * (powf(x, y));
  } else {
    return 0;
  }
}

__device__ float Atan24GpuFloat(float x, float y) { return atan2(x, y); }

__device__ float Atan2CalYDiff4GpuFloat(float x, float y, float dz) {
  return dz * -x / (y * y + x * x);
}

__device__ float Atan2CalXDiff4GpuFloat(float x, float y, float dz) {
  return dz * (y / (x * x + y * y));
}

__device__ float Xdivy4GpuFloat(float x, float y) {
  if (0 == x) {
    return 0;
  } else {
    return x / y;
  }
}

__device__ float XdivyCalXDiff4GpuFloat(float x, float y, float dz) {
  if (0 == x) {
    return 0;
  } else {
    return Xdivy4GpuFloat(dz, y);
  }
}

__device__ float XdivyCalYDiff4GpuFloat(float x, float y, float dz) {
  return dz * Xdivy4GpuFloat((-x), powf(y, 2));
}

__device__ float Xlogy4GpuFloat(float x, float y) {
  if (0 == x) {
    return 0;
  } else {
    return x * logf(y);
  }
}
__device__ float XlogyCalXDiff4GpuFloat(float x, float y, float dz) {
  if (0 == x) {
    return 0;
  } else {
    return Xlogy4GpuFloat(dz, y);
  }
}

__device__ float XlogyCalYDiff4GpuFloat(float x, float y, float dz) {
  return dz * Xdivy4GpuFloat(x, y);
}
__device__ float FloordivFuc(float x, float y) { return floor(fdividef(x, y)); }

__device__ float FloordivCalXDiff4GpuFloat(float x, float y, float dz) { return 0; }

__device__ float FloordivCalYDiff4GpuFloat(float x, float y, float dz) { return 0; }

#define MATH_BINARY_GPU(func_name, fw_func, bw_func_cal_x_diff, bw_func_cal_y_diff, dtype)       \
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

#define MATH_BINARY_GPU_FLOAT_SEQ            \
  OF_PP_MAKE_TUPLE_SEQ("Pow", Pow)           \
  OF_PP_MAKE_TUPLE_SEQ("Atan2", Atan2)       \
  OF_PP_MAKE_TUPLE_SEQ("Floordiv", Floordiv) \
  OF_PP_MAKE_TUPLE_SEQ("Xdivy", Xdivy)       \
  OF_PP_MAKE_TUPLE_SEQ("Xlogy", Xlogy)

MATH_BINARY_GPU(Pow, powf, PowCalXDiff4GpuFloat, PowCalYDiff4GpuFloat, float);
MATH_BINARY_GPU(Atan2, Atan24GpuFloat, Atan2CalXDiff4GpuFloat, Atan2CalYDiff4GpuFloat, float);
MATH_BINARY_GPU(Floordiv, FloordivFuc, FloordivCalXDiff4GpuFloat, FloordivCalYDiff4GpuFloat, float);
MATH_BINARY_GPU(Xdivy, Xdivy4GpuFloat, XdivyCalXDiff4GpuFloat, XdivyCalYDiff4GpuFloat, float);
MATH_BINARY_GPU(Xlogy, Xlogy4GpuFloat, XlogyCalXDiff4GpuFloat, XlogyCalYDiff4GpuFloat, float);

class MathBinaryGpuFloatKernel final : public OpKernel {
 public:
  MathBinaryGpuFloatKernel() = default;
  ~MathBinaryGpuFloatKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
    std::string binary_math_type = ctx->GetAttr<std::string>("binary_math_type");
    bool is_find = false;

#define MATH_BINARY_FORWARD(binary_math_type_str, func_name_prefix)             \
  if (binary_math_type == binary_math_type_str) {                               \
    is_find = true;                                                             \
    func_name_prefix##Forward(ctx->device_ctx(), tensor_x, tensor_y, tensor_z); \
  }

    OF_PP_FOR_EACH_TUPLE(MATH_BINARY_FORWARD, MATH_BINARY_GPU_FLOAT_SEQ);
    CHECK(is_find);

#undef MATH_BINARY_FORWARD
  }
};

REGISTER_USER_KERNEL("binary").SetCreateFn<MathBinaryGpuFloatKernel>().SetIsMatchedPred(
    [](const KernelRegContext& ctx) {
      const user_op::TensorDesc* x_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);
      const user_op::TensorDesc* y_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);
      const user_op::TensorDesc* z_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("z", 0);

      if (ctx.device_type() == DeviceType::kGPU && x_tensor_desc->data_type() == DataType::kFloat
          && y_tensor_desc->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
    });

class MathBinaryXGradGpuFloatKernel final : public OpKernel {
 public:
  MathBinaryXGradGpuFloatKernel() = default;
  ~MathBinaryXGradGpuFloatKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    std::string binary_math_type = ctx->GetAttr<std::string>("binary_math_type");
    bool is_find = false;

#define MATH_BINARY_BACKWARD(binary_math_type_str, func_name_prefix)                          \
  if (binary_math_type == binary_math_type_str) {                                             \
    is_find = true;                                                                           \
    func_name_prefix##XBackward(ctx->device_ctx(), tensor_x, tensor_y, tensor_dz, tensor_dx); \
  }

    OF_PP_FOR_EACH_TUPLE(MATH_BINARY_BACKWARD, MATH_BINARY_GPU_FLOAT_SEQ);
    CHECK(is_find);

#undef MATH_BINARY_FORWARD
  }
};

class MathBinaryYGradGpuFloatKernel final : public OpKernel {
 public:
  MathBinaryYGradGpuFloatKernel() = default;
  ~MathBinaryYGradGpuFloatKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    std::string binary_math_type = ctx->GetAttr<std::string>("binary_math_type");
    bool is_find = false;

#define MATH_BINARY_BACKWARD(binary_math_type_str, func_name_prefix)                          \
  if (binary_math_type == binary_math_type_str) {                                             \
    is_find = true;                                                                           \
    func_name_prefix##YBackward(ctx->device_ctx(), tensor_x, tensor_y, tensor_dz, tensor_dy); \
  }

    OF_PP_FOR_EACH_TUPLE(MATH_BINARY_BACKWARD, MATH_BINARY_GPU_FLOAT_SEQ);
    CHECK(is_find);

#undef MATH_BINARY_FORWARD
  }
};

REGISTER_USER_KERNEL("binary_x_grad")
    .SetCreateFn<MathBinaryXGradGpuFloatKernel>()
    .SetIsMatchedPred([](const KernelRegContext& ctx) {
      const user_op::TensorDesc* x_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);
      if (ctx.device_type() == DeviceType::kGPU && x_tensor_desc->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
    });

REGISTER_USER_KERNEL("binary_y_grad")
    .SetCreateFn<MathBinaryYGradGpuFloatKernel>()
    .SetIsMatchedPred([](const KernelRegContext& ctx) {
      const user_op::TensorDesc* y_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);
      if (ctx.device_type() == DeviceType::kGPU && y_tensor_desc->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
    });

#endif  // WITH_CUDA

}  // namespace user_op

}  // namespace oneflow
