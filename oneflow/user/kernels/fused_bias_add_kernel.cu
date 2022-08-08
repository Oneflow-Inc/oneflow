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
#include "oneflow/core/ep/cuda/cuda_stream.h"
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"

namespace oneflow {

namespace {

template<typename T>
struct GeluFunctor {
  __device__ T Compute(T x, int64_t i) const {
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

#if CUDA_VERSION >= 11000
template<>
struct GeluFunctor<nv_bfloat16> {
  GeluFunctor<float> float_functor;
  __device__ nv_bfloat16 Compute(nv_bfloat16 x, int64_t i) const {
    return static_cast<nv_bfloat16>(float_functor.Compute(static_cast<float>(x), i));
  }
};
#endif

template<typename T>
struct MaskAndScaleFunctor {
  MaskAndScaleFunctor(const bool* mask, float scale) : mask(mask), scale(scale) {}
  __device__ T Compute(T x, int64_t i) const { return x * static_cast<T>(mask[i] * scale); }
  const bool* mask;
  float scale;
};

template<>
struct MaskAndScaleFunctor<half> {
  MaskAndScaleFunctor(const bool* mask, float scale) : mask(mask), scale(scale) {}
  __device__ half Compute(half x, int64_t i) const {
    return x * static_cast<half>(mask[i] * scale);
  }
  __device__ half2 ComputeHalf2(half2 x, int64_t i) const {
    const char2* mask_c2 = reinterpret_cast<const char2*>(mask);
    char2 mask_val = mask_c2[i];
    half2 one_or_zero_h2;
    half2 h2_scale = __float2half2_rn(scale);
    one_or_zero_h2.x = mask_val.x;
    one_or_zero_h2.y = mask_val.y;
    return __hmul2(__hmul2(x, one_or_zero_h2), h2_scale);
  }
  const bool* mask;
  float scale;
};

template<typename T>
struct MaskAndScaleAddFunctor {
  MaskAndScaleAddFunctor(const bool* mask, const T* addend, float scale)
      : mask(mask), addend(addend), scale(scale) {}
  __device__ T Compute(T x, int64_t i) const {
    return x * static_cast<T>(mask[i] * scale) + addend[i];
  }
  const bool* mask;
  const T* addend;
  float scale;
};

template<>
struct MaskAndScaleAddFunctor<half> {
  MaskAndScaleAddFunctor(const bool* mask, const half* addend, float scale)
      : mask(mask), addend(addend), scale(scale) {}
  __device__ half Compute(half x, int64_t i) const {
    return x * static_cast<half>(mask[i] * scale) + addend[i];
  }
  __device__ half2 ComputeHalf2(half2 x, int64_t i) const {
    const char2* mask_c2 = reinterpret_cast<const char2*>(mask);
    const half2* addend_h2 = reinterpret_cast<const half2*>(addend);
    char2 mask_val = mask_c2[i];
    half2 one_or_zero_h2;
    half2 h2_scale = __float2half2_rn(scale);
    one_or_zero_h2.x = mask_val.x;
    one_or_zero_h2.y = mask_val.y;
    return __hadd2(__hmul2(__hmul2(x, one_or_zero_h2), h2_scale), addend_h2[i]);
  }
  const bool* mask;
  const half* addend;
  float scale;
};

template<typename T>
struct GeluGradFunctor {
  const T coef = std::sqrt(static_cast<T>(2.0) / std::acos(static_cast<T>(-1.0)));
  __device__ T Compute(T x, T dy, int64_t i) const {
    return static_cast<T>(0.5)
           * (static_cast<T>(1.0) + erf(static_cast<T>(M_SQRT1_2) * x)
              + x * coef * exp(static_cast<T>(-0.5) * x * x))
           * dy;
  }
};

template<>
struct GeluGradFunctor<half> {
  GeluGradFunctor<float> float_functor;
  __device__ half Compute(half x, half dy, int64_t i) const {
    return __float2half(float_functor.Compute(__half2float(x), __half2float(dy), i));
  }
};

#if CUDA_VERSION >= 11000
template<>
struct GeluGradFunctor<nv_bfloat16> {
  GeluGradFunctor<float> float_functor;
  __device__ nv_bfloat16 Compute(nv_bfloat16 x, nv_bfloat16 dy, int64_t i) const {
    return static_cast<nv_bfloat16>(
        float_functor.Compute(static_cast<float>(x), static_cast<float>(dy), i));
  }
};
#endif

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

template<typename FUNCTOR, typename T, typename Index>
struct FusedBiasAddRow {
  static void Invoke(ep::Stream* stream, FUNCTOR functor, Index elem_cnt, Index bias_size,
                     const T* x, const T* bias, T* y) {
    FusedBiasAddRowGpu<FUNCTOR, T, Index>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(functor, elem_cnt, bias_size, x, bias, y);
  }
};

template<typename FUNCTOR, typename Index>
struct FusedBiasAddRow<FUNCTOR, half, Index> {
  static void Invoke(ep::Stream* stream, FUNCTOR functor, Index elem_cnt, Index bias_size,
                     const half* x, const half* bias, half* y) {
    if (bias_size % 2 == 0) {
      FusedBiasAddRowGpuHalf2<FUNCTOR, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt / 2), kCudaThreadsNumPerBlock, 0,
             stream->As<ep::CudaStream>()->cuda_stream()>>>(functor, elem_cnt, bias_size, x, bias,
                                                            y);
    } else {
      FusedBiasAddRowGpu<FUNCTOR, half, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
             stream->As<ep::CudaStream>()->cuda_stream()>>>(functor, elem_cnt, bias_size, x, bias,
                                                            y);
    }
  }
};

template<typename FUNCTOR, typename T, typename Index>
void FusedBiasAddForwardImpl(ep::Stream* stream, FUNCTOR functor, Index outer_size, Index bias_size,
                             Index inner_size, const T* x, const T* bias, T* y) {
  const Index elem_cnt = outer_size * bias_size * inner_size;
  if (inner_size == 1) {
    FusedBiasAddRow<FUNCTOR, T, Index>::Invoke(stream, functor, elem_cnt, bias_size, x, bias, y);
  } else if (outer_size == 1) {
    FusedBiasAddColGpu<FUNCTOR, T, Index><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock,
                                            0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
        functor, elem_cnt, inner_size, x, bias, y);
  } else {
    FusedBiasAddGpu<FUNCTOR, T, Index><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                         stream->As<ep::CudaStream>()->cuda_stream()>>>(
        functor, elem_cnt, bias_size, inner_size, x, bias, y);
  }
}

template<typename FUNCTOR, typename T, typename Index>
struct FusedBiasAddGradRow {
  static void Invoke(ep::Stream* stream, FUNCTOR grad_functor, Index elem_cnt, Index bias_size,
                     const T* x, const T* bias, const T* dy, T* dx) {
    FusedBiasAddGradRowGpu<FUNCTOR, T, Index>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(grad_functor, elem_cnt, bias_size, x,
                                                          bias, dy, dx);
  }
};

template<typename FUNCTOR, typename Index>
struct FusedBiasAddGradRow<FUNCTOR, half, Index> {
  static void Invoke(ep::Stream* stream, FUNCTOR grad_functor, Index elem_cnt, Index bias_size,
                     const half* x, const half* bias, const half* dy, half* dx) {
    if (bias_size % 2 == 0) {
      FusedBiasAddGradRowGpuHalf2<FUNCTOR, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt / 2), kCudaThreadsNumPerBlock, 0,
             stream->As<ep::CudaStream>()->cuda_stream()>>>(grad_functor, elem_cnt, bias_size, x,
                                                            bias, dy, dx);
    } else {
      FusedBiasAddGradRowGpu<FUNCTOR, half, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
             stream->As<ep::CudaStream>()->cuda_stream()>>>(grad_functor, elem_cnt, bias_size, x,
                                                            bias, dy, dx);
    }
  }
};

template<typename FUNCTOR, typename T, typename Index>
void FusedBiasAddGradImpl(ep::Stream* stream, FUNCTOR grad_functor, Index outer_size,
                          Index bias_size, Index inner_size, const T* x, const T* bias, const T* dy,
                          T* dx) {
  const Index elem_cnt = outer_size * bias_size * inner_size;
  if (inner_size == 1) {
    FusedBiasAddGradRow<FUNCTOR, T, Index>::Invoke(stream, grad_functor, elem_cnt, bias_size, x,
                                                   bias, dy, dx);
  } else if (outer_size == 1) {
    FusedBiasAddGradColGpu<FUNCTOR, T, Index>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(grad_functor, elem_cnt, inner_size, x,
                                                          bias, dy, dx);
  } else {
    FusedBiasAddGradGpu<FUNCTOR, T, Index>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(grad_functor, elem_cnt, bias_size,
                                                          inner_size, x, bias, dy, dx);
  }
}

template<typename FUNCTOR, typename T>
void DispatchFusedBiasAddForwardImpl(ep::Stream* stream, FUNCTOR functor, int64_t n,
                                     int64_t outer_size, int64_t bias_size, int64_t inner_size,
                                     const T* x, const T* bias, T* y) {
  if (IsKernelSafeInt32(n)) {
    FusedBiasAddForwardImpl<FUNCTOR, T, int32_t>(stream, functor, outer_size, bias_size, inner_size,
                                                 x, bias, y);
  } else {
    FusedBiasAddForwardImpl<FUNCTOR, T, int64_t>(stream, functor, outer_size, bias_size, inner_size,
                                                 x, bias, y);
  }
}

}  // namespace

template<typename T>
class FusedFusedBiasAddKernel final : public user_op::OpKernel {
 public:
  FusedFusedBiasAddKernel() = default;
  ~FusedFusedBiasAddKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* a_tensor = ctx->Tensor4ArgNameAndIndex("a", 0);
    const auto* b_tensor = ctx->Tensor4ArgNameAndIndex("b", 0);
    auto* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t bias_add_axis = ctx->Attr<int32_t>("axis");
    const int64_t outer_size = a_tensor->shape_view().Count(0, bias_add_axis);
    const int64_t bias_size = a_tensor->shape_view().At(bias_add_axis);
    const int64_t inner_size = a_tensor->shape_view().Count(bias_add_axis + 1);
    const auto n = a_tensor->shape_view().elem_cnt();
    GeluFunctor<T> gelu_functor{};
    DispatchFusedBiasAddForwardImpl<decltype(gelu_functor), T>(
        ctx->stream(), gelu_functor, n, outer_size, bias_size, inner_size, a_tensor->dptr<T>(),
        b_tensor->dptr<T>(), out_tensor->mut_dptr<T>());
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_BIAS_ADD_GELU_KERNEL(dtype)                     \
  REGISTER_USER_KERNEL("fused_bias_add_gelu")                          \
      .SetCreateFn<FusedFusedBiasAddKernel<dtype>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_BIAS_ADD_GELU_KERNEL(float)
REGISTER_FUSED_BIAS_ADD_GELU_KERNEL(double)
REGISTER_FUSED_BIAS_ADD_GELU_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_FUSED_BIAS_ADD_GELU_KERNEL(nv_bfloat16)
#endif

template<typename T>
class FusedBiasAddMaskScaleKernel final : public user_op::OpKernel {
 public:
  FusedBiasAddMaskScaleKernel() = default;
  ~FusedBiasAddMaskScaleKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* a_tensor = ctx->Tensor4ArgNameAndIndex("a", 0);
    const auto* b_tensor = ctx->Tensor4ArgNameAndIndex("b", 0);
    const auto* mask_tensor = ctx->Tensor4ArgNameAndIndex("mask", 0);
    auto* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t bias_add_axis = ctx->Attr<int32_t>("axis");
    const float scale = ctx->Attr<float>("scale");
    const int64_t outer_size = a_tensor->shape_view().Count(0, bias_add_axis);
    const int64_t bias_size = a_tensor->shape_view().At(bias_add_axis);
    const int64_t inner_size = a_tensor->shape_view().Count(bias_add_axis + 1);
    const auto n = a_tensor->shape_view().elem_cnt();
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* addend = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      MaskAndScaleAddFunctor<T> mask_and_scale_add_functor(mask_tensor->dptr<bool>(),
                                                           addend->dptr<T>(), scale);
      DispatchFusedBiasAddForwardImpl<decltype(mask_and_scale_add_functor), T>(
          ctx->stream(), mask_and_scale_add_functor, n, outer_size, bias_size, inner_size,
          a_tensor->dptr<T>(), b_tensor->dptr<T>(), out_tensor->mut_dptr<T>());
    } else {
      MaskAndScaleFunctor<T> mask_and_scale_functor(mask_tensor->dptr<bool>(), scale);
      DispatchFusedBiasAddForwardImpl<decltype(mask_and_scale_functor), T>(
          ctx->stream(), mask_and_scale_functor, n, outer_size, bias_size, inner_size,
          a_tensor->dptr<T>(), b_tensor->dptr<T>(), out_tensor->mut_dptr<T>());
    }
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_BIAS_ADD_MASK_SCALE_KERNEL(dtype)               \
  REGISTER_USER_KERNEL("fused_bias_add_mask_scale")                    \
      .SetCreateFn<FusedBiasAddMaskScaleKernel<dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_BIAS_ADD_MASK_SCALE_KERNEL(float)
REGISTER_FUSED_BIAS_ADD_MASK_SCALE_KERNEL(double)
REGISTER_FUSED_BIAS_ADD_MASK_SCALE_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_FUSED_BIAS_ADD_MASK_SCALE_KERNEL(nv_bfloat16)
#endif

template<typename T>
class FusedFusedBiasAddGradKernel final : public user_op::OpKernel {
 public:
  FusedFusedBiasAddGradKernel() = default;
  ~FusedFusedBiasAddGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* a_tensor = ctx->Tensor4ArgNameAndIndex("a", 0);
    const auto* b_tensor = ctx->Tensor4ArgNameAndIndex("b", 0);
    const auto* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t bias_add_axis = ctx->Attr<int32_t>("axis");
    const int64_t outer_size = a_tensor->shape_view().Count(0, bias_add_axis);
    const int64_t bias_size = a_tensor->shape_view().At(bias_add_axis);
    const int64_t inner_size = a_tensor->shape_view().Count(bias_add_axis + 1);
    const auto n = a_tensor->shape_view().elem_cnt();
    GeluGradFunctor<T> gelu_grad_functor;
    if (IsKernelSafeInt32(n)) {
      FusedBiasAddGradImpl<decltype(gelu_grad_functor), T, int32_t>(
          ctx->stream(), gelu_grad_functor, outer_size, bias_size, inner_size, a_tensor->dptr<T>(),
          b_tensor->dptr<T>(), dy_tensor->dptr<T>(), dx_tensor->mut_dptr<T>());
    } else {
      FusedBiasAddGradImpl<decltype(gelu_grad_functor), T, int64_t>(
          ctx->stream(), gelu_grad_functor, outer_size, bias_size, inner_size, a_tensor->dptr<T>(),
          b_tensor->dptr<T>(), dy_tensor->dptr<T>(), dx_tensor->mut_dptr<T>());
    }
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_BIAS_ADD_GELU_GRAD_KERNEL(dtype)                \
  REGISTER_USER_KERNEL("fused_bias_add_gelu_grad")                     \
      .SetCreateFn<FusedFusedBiasAddGradKernel<dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_BIAS_ADD_GELU_GRAD_KERNEL(float)
REGISTER_FUSED_BIAS_ADD_GELU_GRAD_KERNEL(double)
REGISTER_FUSED_BIAS_ADD_GELU_GRAD_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_FUSED_BIAS_ADD_GELU_GRAD_KERNEL(nv_bfloat16)
#endif

}  // namespace oneflow
