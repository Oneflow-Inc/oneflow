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
  OF_DEVICE_FUNC T operator()(T x) const {
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + erf(static_cast<T>(M_SQRT1_2) * x));
  }
};

template<typename T>
struct GeluGradFunctor {
  const T coef = sqrt(static_cast<T>(2.0) / acos(static_cast<T>(-1.0)));
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    return static_cast<T>(0.5)
           * (static_cast<T>(1.0) + erf(static_cast<T>(M_SQRT1_2) * x)
              + x * coef * exp(static_cast<T>(-0.5) * x * x))
           * dy;
  }
};

template<>
struct GeluFunctor<half> {
  GeluFunctor<float> float_functor;
  OF_DEVICE_FUNC half operator()(half x) const {
    return __float2half(float_functor(__half2float(x)));
  }
};

template<>
struct GeluGradFunctor<half> {
  GeluGradFunctor<float> float_functor;
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
};

template<typename T, typename Index>
__global__ void BiasAddGeluGpu(const Index elem_cnt, const Index bias_size, const Index inner_size,
                               const T* x, const T* bias, T* y) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    T x_i = x[i] + bias[(i % block_size) / inner_size];
    y[i] = GeluFunctor<T>()(x_i);
  }
}

template<typename T, typename Index>
__global__ void BiasAddGeluGradGpu(const Index elem_cnt, const Index bias_size,
                                   const Index inner_size, const T* x, const T* bias, const T* dy,
                                   T* dx) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    T x_i = x[i] + bias[(i % block_size) / inner_size];
    dx[i] = GeluGradFunctor<T>()(x_i, dy[i]);
  }
}

template<typename Index>
__global__ void BiasAddGeluGpuHalf(const Index elem_cnt, const Index bias_size,
                                   const Index inner_size, const half* x, const half* bias,
                                   half* y) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    float x_i = __half2float(x[i]) + __half2float(bias[(i % block_size) / inner_size]);
    y[i] = __float2half(GeluFunctor<float>()(x_i));
  }
}

template<typename Index>
__global__ void BiasAddGeluGradGpuHalf(const Index elem_cnt, const Index bias_size,
                                       const Index inner_size, const half* x, const half* bias,
                                       const half* dy, half* dx) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    float x_i = __half2float(x[i]) + __half2float(bias[(i % block_size) / inner_size]);
    dx[i] = __float2half(GeluGradFunctor<float>()(x_i, __half2float(dy[i])));
  }
}

template<typename T, typename Index>
__global__ void BiasAddGeluRowGpu(const Index elem_cnt, const Index bias_size, const T* x,
                                  const T* bias, T* y) {
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    T x_i = x[i] + bias[i % bias_size];
    y[i] = GeluFunctor<T>()(x_i);
  }
}

template<typename T, typename Index>
__global__ void BiasAddGeluGradRowGpu(const Index elem_cnt, const Index bias_size, const T* x,
                                      const T* bias, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    T x_i = x[i] + bias[i % bias_size];
    dx[i] = GeluGradFunctor<T>()(x_i, dy[i]);
  }
}

template<typename Index>
__global__ void BiasAddGeluRowGpuHalf2(const Index elem_cnt, const Index bias_size, const half* x,
                                       const half* bias, half* y) {
  const Index h2_elem_cnt = elem_cnt / 2;
  const Index h2_bias_size = bias_size / 2;
  const auto* x_h2 = reinterpret_cast<const half2*>(x);
  const auto* bias_h2 = reinterpret_cast<const half2*>(bias);
  auto* y_h2 = reinterpret_cast<half2*>(y);
  CUDA_1D_KERNEL_LOOP_T(Index, i, h2_elem_cnt) {
    float x_i_0 = __half2float(x_h2[i].x) + __half2float(bias_h2[i % h2_bias_size].x);
    float x_i_1 = __half2float(x_h2[i].y) + __half2float(bias_h2[i % h2_bias_size].y);
    float2 y_i;
    y_i.x = GeluFunctor<float>()(x_i_0);
    y_i.y = GeluFunctor<float>()(x_i_1);
    y_h2[i] = __float22half2_rn(y_i);
  }
}

template<typename Index>
__global__ void BiasAddGeluGradRowGpuHalf2(const Index elem_cnt, const Index bias_size,
                                           const half* x, const half* bias, const half* dy,
                                           half* dx) {
  const Index h2_elem_cnt = elem_cnt / 2;
  const Index h2_bias_size = bias_size / 2;
  const auto* x_h2 = reinterpret_cast<const half2*>(x);
  const auto* bias_h2 = reinterpret_cast<const half2*>(bias);
  const auto* dy_h2 = reinterpret_cast<const half2*>(dy);
  auto* dx_h2 = reinterpret_cast<half2*>(dx);
  CUDA_1D_KERNEL_LOOP_T(Index, i, h2_elem_cnt) {
    float x_i_0 = __half2float(x_h2[i].x) + __half2float(bias_h2[i % h2_bias_size].x);
    float x_i_1 = __half2float(x_h2[i].y) + __half2float(bias_h2[i % h2_bias_size].y);
    float2 dy_i = __half2float2(dy_h2[i]);
    float2 dx_i;
    dx_i.x = GeluGradFunctor<float>()(x_i_0, __half2float(dy_i.x));
    dx_i.y = GeluGradFunctor<float>()(x_i_1, __half2float(dy_i.y));
    dx_h2[i] = __float22half2_rn(dx_i);
  }
}

template<typename T, typename Index>
__global__ void BiasAddGeluColGpu(const Index elem_cnt, const Index inner_size, const T* x,
                                  const T* bias, T* y) {
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    T x_i = x[i] + bias[i / inner_size];
    y[i] = GeluFunctor<T>()(x_i);
  }
}

template<typename T, typename Index>
__global__ void BiasAddGeluGradColGpu(const Index elem_cnt, const Index inner_size, const T* x,
                                      const T* bias, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    T x_i = x[i] + bias[i / inner_size];
    dx[i] = GeluGradFunctor<T>()(x_i, dy[i]);
  }
}

}  // namespace

template<typename T, typename Index>
struct BiasAddGeluCalculation {
  static void Invoke(DeviceCtx* ctx, Index outer_size, Index bias_size, Index inner_size,
                     const T* x, const T* bias, T* y) {
    const Index elem_cnt = outer_size * bias_size * inner_size;
    if (inner_size == 1) {
      BiasAddGeluRowGpu<T, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              elem_cnt, bias_size, x, bias, y);
    } else if (outer_size == 1) {
      BiasAddGeluColGpu<T, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              elem_cnt, inner_size, x, bias, y);
    } else {
      RUN_CUDA_KERNEL((BiasAddGeluGpu<T, Index>), ctx, elem_cnt, elem_cnt, bias_size, inner_size, x,
                      bias, y);
    }
  }
};

template<typename Index>
struct BiasAddGeluCalculation<float16, Index> {
  static void Invoke(DeviceCtx* ctx, Index outer_size, Index bias_size, Index inner_size,
                     const float16* x, const float16* bias, float16* y) {
    const Index elem_cnt = outer_size * bias_size * inner_size;
    if (inner_size == 1) {
      if (bias_size % 2 == 0) {
        BiasAddGeluRowGpuHalf2<Index><<<BlocksNum4ThreadsNum(elem_cnt / 2), kCudaThreadsNumPerBlock,
                                        0, ctx->cuda_stream()>>>(
            elem_cnt, bias_size, reinterpret_cast<const half*>(x),
            reinterpret_cast<const half*>(bias), reinterpret_cast<half*>(y));
      } else {
        BiasAddGeluRowGpu<half, Index>
            <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
                elem_cnt, bias_size, reinterpret_cast<const half*>(x),
                reinterpret_cast<const half*>(bias), reinterpret_cast<half*>(y));
      }
    } else if (outer_size == 1) {
      BiasAddGeluColGpu<half, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              elem_cnt, inner_size, reinterpret_cast<const half*>(x),
              reinterpret_cast<const half*>(bias), reinterpret_cast<half*>(y));
    } else {
      RUN_CUDA_KERNEL((BiasAddGeluGpuHalf<Index>), ctx, elem_cnt, elem_cnt, bias_size, inner_size,
                      reinterpret_cast<const half*>(x), reinterpret_cast<const half*>(bias),
                      reinterpret_cast<half*>(y));
    }
  }
};

template<typename T, typename Index>
struct BiasAddGeluGradCalculation {
  static void Invoke(DeviceCtx* ctx, Index outer_size, Index bias_size, Index inner_size,
                     const T* x, const T* bias, const T* dy, T* dx) {
    const Index elem_cnt = outer_size * bias_size * inner_size;
    if (inner_size == 1) {
      BiasAddGeluGradRowGpu<T, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              elem_cnt, bias_size, x, bias, dy, dx);
    } else if (outer_size == 1) {
      BiasAddGeluGradColGpu<T, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              elem_cnt, inner_size, x, bias, dy, dx);
    } else {
      RUN_CUDA_KERNEL((BiasAddGeluGradGpu<T, Index>), ctx, elem_cnt, elem_cnt, bias_size,
                      inner_size, x, bias, dy, dx);
    }
  }
};

template<typename Index>
struct BiasAddGeluGradCalculation<float16, Index> {
  static void Invoke(DeviceCtx* ctx, Index outer_size, Index bias_size, Index inner_size,
                     const float16* x, const float16* bias, const float16* dy, float16* dx) {
    const Index elem_cnt = outer_size * bias_size * inner_size;
    if (inner_size == 1) {
      if (bias_size % 2 == 0) {
        BiasAddGeluGradRowGpuHalf2<Index><<<BlocksNum4ThreadsNum(elem_cnt / 2),
                                            kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, bias_size, reinterpret_cast<const half*>(x),
            reinterpret_cast<const half*>(bias), reinterpret_cast<const half*>(dy),
            reinterpret_cast<half*>(dx));
      } else {
        BiasAddGeluGradRowGpu<half, Index>
            <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
                elem_cnt, bias_size, reinterpret_cast<const half*>(x),
                reinterpret_cast<const half*>(bias), reinterpret_cast<const half*>(dy),
                reinterpret_cast<half*>(dx));
      }
    } else if (outer_size == 1) {
      BiasAddGeluGradColGpu<half, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              elem_cnt, inner_size, reinterpret_cast<const half*>(x),
              reinterpret_cast<const half*>(bias), reinterpret_cast<const half*>(dy),
              reinterpret_cast<half*>(dx));
    } else {
      RUN_CUDA_KERNEL((BiasAddGeluGradGpuHalf<Index>), ctx, elem_cnt, elem_cnt, bias_size,
                      inner_size, reinterpret_cast<const half*>(x),
                      reinterpret_cast<const half*>(bias), reinterpret_cast<const half*>(dy),
                      reinterpret_cast<half*>(dx));
    }
  }
};

template<typename T>
class FusedBiasAddGeluKernel final : public user_op::OpKernel {
 public:
  FusedBiasAddGeluKernel() = default;
  ~FusedBiasAddGeluKernel() override = default;

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
    if (IsKernelSafeInt32(n)) {
      BiasAddGeluCalculation<T, int32_t>::Invoke(ctx->device_ctx(), outer_size, bias_size,
                                                 inner_size, a_tensor->dptr<T>(),
                                                 b_tensor->dptr<T>(), out_tensor->mut_dptr<T>());
    } else {
      BiasAddGeluCalculation<T, int64_t>::Invoke(ctx->device_ctx(), outer_size, bias_size,
                                                 inner_size, a_tensor->dptr<T>(),
                                                 b_tensor->dptr<T>(), out_tensor->mut_dptr<T>());
    }
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_BIAS_ADD_GELU_KERNEL(dtype)        \
  REGISTER_USER_KERNEL("fused_bias_add_gelu")             \
      .SetCreateFn<FusedBiasAddGeluKernel<dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_BIAS_ADD_GELU_KERNEL(float)
REGISTER_FUSED_BIAS_ADD_GELU_KERNEL(double)
REGISTER_FUSED_BIAS_ADD_GELU_KERNEL(half)

template<typename T>
class FusedBiasAddGeluGradKernel final : public user_op::OpKernel {
 public:
  FusedBiasAddGeluGradKernel() = default;
  ~FusedBiasAddGeluGradKernel() override = default;

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
    if (IsKernelSafeInt32(n)) {
      BiasAddGeluGradCalculation<T, int32_t>::Invoke(
          ctx->device_ctx(), outer_size, bias_size, inner_size, a_tensor->dptr<T>(),
          b_tensor->dptr<T>(), dy_tensor->dptr<T>(), dx_tensor->mut_dptr<T>());
    } else {
      BiasAddGeluGradCalculation<T, int64_t>::Invoke(
          ctx->device_ctx(), outer_size, bias_size, inner_size, a_tensor->dptr<T>(),
          b_tensor->dptr<T>(), dy_tensor->dptr<T>(), dx_tensor->mut_dptr<T>());
    }
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_BIAS_ADD_GELU_GRAD_KERNEL(dtype)   \
  REGISTER_USER_KERNEL("fused_bias_add_gelu_grad")        \
      .SetCreateFn<FusedBiasAddGeluGradKernel<dtype>>()   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_BIAS_ADD_GELU_GRAD_KERNEL(float)
REGISTER_FUSED_BIAS_ADD_GELU_GRAD_KERNEL(double)
REGISTER_FUSED_BIAS_ADD_GELU_GRAD_KERNEL(half)

}  // namespace oneflow
