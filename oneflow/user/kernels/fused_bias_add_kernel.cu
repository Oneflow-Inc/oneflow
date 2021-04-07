/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template<typename T>
struct GeluFunctor {
  OF_DEVICE_FUNC T Compute(T x, int64_t i) const {
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + erf(static_cast<T>(M_SQRT1_2) * x));
  }
};

template<>
struct GeluFunctor<half> {
  GeluFunctor<float> float_functor;
  __device__ half Compute(half x, int64_t i) const {
    return __float2half(float_functor.Compute(__half2float(x), i));
  }
  __device__ half2 ComputeHalf2(half2 x, int64_t i) const {
    half2 y;
    y.x = __float2half(float_functor.Compute(__half2float(x.x), 2 * i));
    y.y = __float2half(float_functor.Compute(__half2float(x.y), 2 * i + 1));
    return y;
  }
};

template<typename T>
struct MaskAndScaleFunctor {
  MaskAndScaleFunctor(const int8_t* mask, float scale) : mask(mask), scale(scale) {}
  OF_DEVICE_FUNC T Compute(T x, int64_t i) const { return x * static_cast<T>(mask[i]) * scale; }
  const int8_t* mask;
  float scale;
};

template<>
struct MaskAndScaleFunctor<half> {
  MaskAndScaleFunctor(const int8_t* mask, float scale) : mask(mask), scale(scale) {
    mask_c2 = reinterpret_cast<const char2*>(mask);
  }
  __device__ half Compute(half x, int64_t i) const {
    return x * static_cast<half>(mask[i] * scale);
  }
  __device__ half2 ComputeHalf2(half2 x, int64_t i) const {
    char2 mask_val = mask_c2[i];
    half2 one_or_zero_h2;
    half2 h2_scale = __float2half2_rn(scale);
    one_or_zero_h2.x = mask_val.x;
    one_or_zero_h2.y = mask_val.y;
    return __hmul2(__hmul2(x, one_or_zero_h2), h2_scale);
  }
  const int8_t* mask;
  const char2* mask_c2;
  float scale;
};

template<typename T>
struct MaskAndScaleAddFunctor {
  MaskAndScaleAddFunctor(const int8_t* mask, const T* addend, T scale)
      : mask(mask), addend(addend), scale(scale) {}
  OF_DEVICE_FUNC T Compute(T x, int64_t i) const {
    return x * static_cast<T>(mask[i]) * scale + addend[i];
  }
  const int8_t* mask;
  const T* addend;
  T scale;
};

template<>
struct MaskAndScaleAddFunctor<half> {
  MaskAndScaleAddFunctor(const int8_t* mask, const half* addend, float scale)
      : mask(mask), addend(addend), scale(scale) {
    mask_c2 = reinterpret_cast<const char2*>(mask);
    addend_h2 = reinterpret_cast<const half2*>(addend);
  }
  __device__ half Compute(half x, int64_t i) const {
    return x * static_cast<half>(mask[i] * scale) + addend[i];
  }
  __device__ half2 ComputeHalf2(half2 x, int64_t i) const {
    char2 mask_val = mask_c2[i];
    half2 one_or_zero_h2;
    half2 h2_scale = __float2half2_rn(scale);
    one_or_zero_h2.x = mask_val.x;
    one_or_zero_h2.y = mask_val.y;
    return __hadd2(__hmul2(__hmul2(x, one_or_zero_h2), h2_scale), addend_h2[i]);
  }
  const int8_t* mask;
  const half* addend;
  const char2* mask_c2;
  const half2* addend_h2;
  float scale;
};

template<typename T>
struct GeluGradFunctor {
  const T coef = sqrt(static_cast<T>(2.0) / acos(static_cast<T>(-1.0)));
  OF_DEVICE_FUNC T Compute(T x, T dy, int64_t i) const {
    return static_cast<T>(0.5)
           * (static_cast<T>(1.0) + erf(static_cast<T>(M_SQRT1_2) * x)
              + x * coef * exp(static_cast<T>(-0.5) * x * x))
           * dy;
  }
};

template<>
struct GeluGradFunctor<half> {
  GeluGradFunctor<float> float_functor;
  OF_DEVICE_FUNC half Compute(half x, half dy, int64_t i) const {
    return __float2half(float_functor.Compute(__half2float(x), __half2float(dy), i));
  }
};

template<typename FUNCTOR, typename T, typename Index>
__global__ void FusedBiasAddGpu(FUNCTOR functor, const Index elem_cnt, const Index bias_size,
                                const Index inner_size, const T* x, const T* bias, T* y) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    T x_i = x[i] + bias[(i % block_size) / inner_size];
    y[i] = functor.Compute(x_i, i);
  }
}

template<typename FUNCTOR, typename T, typename Index>
__global__ void FusedBiasAddGradGpu(FUNCTOR grad_functor, const Index elem_cnt,
                                    const Index bias_size, const Index inner_size, const T* x,
                                    const T* bias, const T* dy, T* dx) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    T x_i = x[i] + bias[(i % block_size) / inner_size];
    dx[i] = grad_functor.Compute(x_i, dy[i], i);
  }
}

template<typename FUNCTOR, typename T, typename Index>
__global__ void FusedBiasAddRowGpu(FUNCTOR functor, const Index elem_cnt, const Index bias_size,
                                   const T* x, const T* bias, T* y) {
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    T x_i = x[i] + bias[i % bias_size];
    y[i] = functor.Compute(x_i, i);
  }
}

template<typename FUNCTOR, typename T, typename Index>
__global__ void FusedBiasAddGradRowGpu(FUNCTOR grad_functor, const Index elem_cnt,
                                       const Index bias_size, const T* x, const T* bias,
                                       const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    T x_i = x[i] + bias[i % bias_size];
    dx[i] = grad_functor.Compute(x_i, dy[i], i);
  }
}

template<typename FUNCTOR, typename Index>
__global__ void FusedBiasAddRowGpuHalf2(FUNCTOR functor, const Index elem_cnt,
                                        const Index bias_size, const half* x, const half* bias,
                                        half* y) {
  const Index h2_elem_cnt = elem_cnt / 2;
  const Index h2_bias_size = bias_size / 2;
  const auto* x_h2 = reinterpret_cast<const half2*>(x);
  const auto* bias_h2 = reinterpret_cast<const half2*>(bias);
  auto* y_h2 = reinterpret_cast<half2*>(y);
  CUDA_1D_KERNEL_LOOP_T(Index, i, h2_elem_cnt) {
    half2 x_i = __hadd2(x_h2[i], bias_h2[i % h2_bias_size]);
    y_h2[i] = functor.ComputeHalf2(x_i, i);
  }
}

template<typename FUNCTOR, typename Index>
__global__ void FusedBiasAddGradRowGpuHalf2(FUNCTOR grad_functor, const Index elem_cnt,
                                            const Index bias_size, const half* x, const half* bias,
                                            const half* dy, half* dx) {
  const Index h2_elem_cnt = elem_cnt / 2;
  const Index h2_bias_size = bias_size / 2;
  const auto* x_h2 = reinterpret_cast<const half2*>(x);
  const auto* bias_h2 = reinterpret_cast<const half2*>(bias);
  const auto* dy_h2 = reinterpret_cast<const half2*>(dy);
  auto* dx_h2 = reinterpret_cast<half2*>(dx);
  CUDA_1D_KERNEL_LOOP_T(Index, i, h2_elem_cnt) {
    half2 x_i = __hadd2(x_h2[i], bias_h2[i % h2_bias_size]);
    half2 dy_i = dy_h2[i];
    half2 dx_i;
    dx_i.x = grad_functor.Compute(x_i.x, dy_i.x, 2 * i);
    dx_i.y = grad_functor.Compute(x_i.y, dy_i.y, 2 * i + 1);
    dx_h2[i] = dx_i;
  }
}

template<typename FUNCTOR, typename T, typename Index>
__global__ void FusedBiasAddColGpu(FUNCTOR functor, const Index elem_cnt, const Index inner_size,
                                   const T* x, const T* bias, T* y) {
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    T x_i = x[i] + bias[i / inner_size];
    y[i] = functor.Compute(x_i, i);
  }
}

template<typename FUNCTOR, typename T, typename Index>
__global__ void FusedBiasAddGradColGpu(FUNCTOR grad_functor, const Index elem_cnt,
                                       const Index inner_size, const T* x, const T* bias,
                                       const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    T x_i = x[i] + bias[i / inner_size];
    dx[i] = grad_functor.Compute(x_i, dy[i], i);
  }
}

}  // namespace

template<typename FUNCTOR, typename T, typename Index>
struct FusedBiasAddRowCalculation {
  static void Invoke(DeviceCtx* ctx, FUNCTOR functor, Index elem_cnt, Index bias_size, const T* x,
                     const T* bias, T* y) {
    FusedBiasAddRowGpu<FUNCTOR, T, Index>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            functor, elem_cnt, bias_size, x, bias, y);
  }
};

template<typename FUNCTOR, typename Index>
struct FusedBiasAddRowCalculation<FUNCTOR, half, Index> {
  static void Invoke(DeviceCtx* ctx, FUNCTOR functor, Index elem_cnt, Index bias_size,
                     const half* x, const half* bias, half* y) {
    if (bias_size % 2 == 0) {
      FusedBiasAddRowGpuHalf2<FUNCTOR, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt / 2), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              functor, elem_cnt, bias_size, x, bias, y);
    } else {
      FusedBiasAddRowGpu<FUNCTOR, half, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              functor, elem_cnt, bias_size, x, bias, y);
    }
  }
};

template<typename FUNCTOR, typename T, typename Index>
void FusedBiasAddCalculation(DeviceCtx* ctx, FUNCTOR functor, Index outer_size, Index bias_size,
                             Index inner_size, const T* x, const T* bias, T* y) {
  const Index elem_cnt = outer_size * bias_size * inner_size;
  if (inner_size == 1) {
    FusedBiasAddRowCalculation<FUNCTOR, T, Index>::Invoke(ctx, functor, elem_cnt, bias_size, x,
                                                          bias, y);
  } else if (outer_size == 1) {
    FusedBiasAddColGpu<FUNCTOR, T, Index>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            functor, elem_cnt, inner_size, x, bias, y);
  } else {
    RUN_CUDA_KERNEL((FusedBiasAddGpu<FUNCTOR, T, Index>), ctx, elem_cnt, functor, elem_cnt,
                    bias_size, inner_size, x, bias, y);
  }
}

template<typename FUNCTOR, typename T, typename Index>
struct FusedBiasAddGradRowCalculation {
  static void Invoke(DeviceCtx* ctx, FUNCTOR grad_functor, Index elem_cnt, Index bias_size,
                     const T* x, const T* bias, const T* dy, T* dx) {
    FusedBiasAddGradRowGpu<FUNCTOR, T, Index>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            grad_functor, elem_cnt, bias_size, x, bias, dy, dx);
  }
};

template<typename FUNCTOR, typename Index>
struct FusedBiasAddGradRowCalculation<FUNCTOR, half, Index> {
  static void Invoke(DeviceCtx* ctx, FUNCTOR grad_functor, Index elem_cnt, Index bias_size,
                     const half* x, const half* bias, const half* dy, half* dx) {
    if (bias_size % 2 == 0) {
      FusedBiasAddGradRowGpuHalf2<FUNCTOR, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt / 2), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              grad_functor, elem_cnt, bias_size, x, bias, dy, dx);
    } else {
      FusedBiasAddGradRowGpu<FUNCTOR, half, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              grad_functor, elem_cnt, bias_size, x, bias, dy, dx);
    }
  }
};

template<typename FUNCTOR, typename T, typename Index>
void FusedBiasAddGradCalculation(DeviceCtx* ctx, FUNCTOR grad_functor, Index outer_size,
                                 Index bias_size, Index inner_size, const T* x, const T* bias,
                                 const T* dy, T* dx) {
  const Index elem_cnt = outer_size * bias_size * inner_size;
  if (inner_size == 1) {
    FusedBiasAddGradRowCalculation<FUNCTOR, T, Index>::Invoke(ctx, grad_functor, elem_cnt,
                                                              bias_size, x, bias, dy, dx);
  } else if (outer_size == 1) {
    FusedBiasAddGradColGpu<FUNCTOR, T, Index>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            grad_functor, elem_cnt, inner_size, x, bias, dy, dx);
  } else {
    RUN_CUDA_KERNEL((FusedBiasAddGradGpu<FUNCTOR, T, Index>), ctx, elem_cnt, grad_functor, elem_cnt,
                    bias_size, inner_size, x, bias, dy, dx);
  }
}

template<typename FUNCTOR, typename T>
void DispatchFusedBiasAddCalculation(DeviceCtx* ctx, FUNCTOR functor, int64_t n, int64_t outer_size,
                                     int64_t bias_size, int64_t inner_size, const T* x,
                                     const T* bias, T* y) {
  if (IsKernelSafeInt32(n)) {
    FusedBiasAddCalculation<FUNCTOR, T, int32_t>(ctx, functor, outer_size, bias_size, inner_size, x,
                                                 bias, y);
  } else {
    FusedBiasAddCalculation<FUNCTOR, T, int64_t>(ctx, functor, outer_size, bias_size, inner_size, x,
                                                 bias, y);
  }
}

template<typename T>
class FusedFusedBiasAddKernel final : public user_op::OpKernel {
 public:
  FusedFusedBiasAddKernel() = default;
  ~FusedFusedBiasAddKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* a_tensor = ctx->Tensor4ArgNameAndIndex("a", 0);
    const auto* b_tensor = ctx->Tensor4ArgNameAndIndex("b", 0);
    auto* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t bias_add_axis = ctx->Attr<int32_t>("axis");
    const int64_t outer_size = a_tensor->shape().Count(0, bias_add_axis);
    const int64_t bias_size = a_tensor->shape().At(bias_add_axis);
    const int64_t inner_size = a_tensor->shape().Count(bias_add_axis + 1);
    const auto n = a_tensor->shape().elem_cnt();
    GeluFunctor<T> gelu_functor;
    DispatchFusedBiasAddCalculation<decltype(gelu_functor), T>(
        ctx->device_ctx(), gelu_functor, n, outer_size, bias_size, inner_size, a_tensor->dptr<T>(),
        b_tensor->dptr<T>(), out_tensor->mut_dptr<T>());
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_BIAS_ADD_GELU_KERNEL(dtype)        \
  REGISTER_USER_KERNEL("fused_bias_add_gelu")             \
      .SetCreateFn<FusedFusedBiasAddKernel<dtype>>()      \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_BIAS_ADD_GELU_KERNEL(float)
REGISTER_FUSED_BIAS_ADD_GELU_KERNEL(double)
REGISTER_FUSED_BIAS_ADD_GELU_KERNEL(half)

template<typename T>
class FusedBiasAddMaskScaleKernel final : public user_op::OpKernel {
 public:
  FusedBiasAddMaskScaleKernel() = default;
  ~FusedBiasAddMaskScaleKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* a_tensor = ctx->Tensor4ArgNameAndIndex("a", 0);
    const auto* b_tensor = ctx->Tensor4ArgNameAndIndex("b", 0);
    const auto* mask_tensor = ctx->Tensor4ArgNameAndIndex("mask", 0);
    auto* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t bias_add_axis = ctx->Attr<int32_t>("axis");
    const float scale = ctx->Attr<float>("scale");
    const int64_t outer_size = a_tensor->shape().Count(0, bias_add_axis);
    const int64_t bias_size = a_tensor->shape().At(bias_add_axis);
    const int64_t inner_size = a_tensor->shape().Count(bias_add_axis + 1);
    const auto n = a_tensor->shape().elem_cnt();
    if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
      const user_op::Tensor* addend = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      MaskAndScaleAddFunctor<T> mask_and_scale_add_functor(mask_tensor->dptr<int8_t>(),
                                                           addend->dptr<T>(), scale);
      DispatchFusedBiasAddCalculation<decltype(mask_and_scale_add_functor), T>(
          ctx->device_ctx(), mask_and_scale_add_functor, n, outer_size, bias_size, inner_size,
          a_tensor->dptr<T>(), b_tensor->dptr<T>(), out_tensor->mut_dptr<T>());
    } else {
      MaskAndScaleFunctor<T> mask_and_scale_functor(mask_tensor->dptr<int8_t>(), scale);
      DispatchFusedBiasAddCalculation<decltype(mask_and_scale_functor), T>(
          ctx->device_ctx(), mask_and_scale_functor, n, outer_size, bias_size, inner_size,
          a_tensor->dptr<T>(), b_tensor->dptr<T>(), out_tensor->mut_dptr<T>());
    }
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_BIAS_ADD_MASK_SCALE_KERNEL(dtype)  \
  REGISTER_USER_KERNEL("fused_bias_add_mask_scale")       \
      .SetCreateFn<FusedBiasAddMaskScaleKernel<dtype>>()  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_BIAS_ADD_MASK_SCALE_KERNEL(float)
REGISTER_FUSED_BIAS_ADD_MASK_SCALE_KERNEL(double)
REGISTER_FUSED_BIAS_ADD_MASK_SCALE_KERNEL(half)

template<typename T>
class FusedFusedBiasAddGradKernel final : public user_op::OpKernel {
 public:
  FusedFusedBiasAddGradKernel() = default;
  ~FusedFusedBiasAddGradKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* a_tensor = ctx->Tensor4ArgNameAndIndex("a", 0);
    const auto* b_tensor = ctx->Tensor4ArgNameAndIndex("b", 0);
    const auto* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t bias_add_axis = ctx->Attr<int32_t>("axis");
    const int64_t outer_size = a_tensor->shape().Count(0, bias_add_axis);
    const int64_t bias_size = a_tensor->shape().At(bias_add_axis);
    const int64_t inner_size = a_tensor->shape().Count(bias_add_axis + 1);
    const auto n = a_tensor->shape().elem_cnt();
    GeluGradFunctor<T> gelu_grad_functor;
    if (IsKernelSafeInt32(n)) {
      FusedBiasAddGradCalculation<decltype(gelu_grad_functor), T, int32_t>(
          ctx->device_ctx(), gelu_grad_functor, outer_size, bias_size, inner_size,
          a_tensor->dptr<T>(), b_tensor->dptr<T>(), dy_tensor->dptr<T>(), dx_tensor->mut_dptr<T>());
    } else {
      FusedBiasAddGradCalculation<decltype(gelu_grad_functor), T, int64_t>(
          ctx->device_ctx(), gelu_grad_functor, outer_size, bias_size, inner_size,
          a_tensor->dptr<T>(), b_tensor->dptr<T>(), dy_tensor->dptr<T>(), dx_tensor->mut_dptr<T>());
    }
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_BIAS_ADD_GELU_GRAD_KERNEL(dtype)   \
  REGISTER_USER_KERNEL("fused_bias_add_gelu_grad")        \
      .SetCreateFn<FusedFusedBiasAddGradKernel<dtype>>()  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_BIAS_ADD_GELU_GRAD_KERNEL(float)
REGISTER_FUSED_BIAS_ADD_GELU_GRAD_KERNEL(double)
REGISTER_FUSED_BIAS_ADD_GELU_GRAD_KERNEL(half)

}  // namespace oneflow
