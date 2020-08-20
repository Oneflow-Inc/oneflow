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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"

namespace oneflow {

namespace {

__device__ float fgelu_forward(float x, float inv_sqrt2) {
  return 0.5f * x * (1.0f + erff(inv_sqrt2 * x));
}

__device__ float fgelu_backward(float x, float dy, float inv_sqrt2, float coef) {
  return 0.5f * (1.0f + erff(inv_sqrt2 * x) + x * coef * expf(-0.5f * x * x)) * dy;
}

template<typename T>
__global__ void GeluForwardGpu(const int64_t n, const T* x, const T inv_sqrt2, T* y) {
  UNIMPLEMENTED();
}

template<typename T>
__global__ void GeluBackwardGpu(const int64_t n, const T* x, const T* dy, const T inv_sqrt2,
                                const T coef, T* dx) {
  UNIMPLEMENTED();
}

template<>
__global__ void GeluForwardGpu(const int64_t n, const float* x, const float inv_sqrt2, float* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = fgelu_forward(x[i], inv_sqrt2); }
}

template<>
__global__ void GeluBackwardGpu(const int64_t n, const float* x, const float* dy,
                                const float inv_sqrt2, const float coef, float* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = fgelu_backward(x[i], dy[i], inv_sqrt2, coef); }
}

template<>
__global__ void GeluForwardGpu(const int64_t n, const double* x, const double inv_sqrt2,
                               double* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 0.5 * x[i] * (1.0 + erf(inv_sqrt2 * x[i])); }
}

template<>
__global__ void GeluBackwardGpu(const int64_t n, const double* x, const double* dy,
                                const double inv_sqrt2, const double coef, double* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    dx[i] = 0.5 * (1.0 + erf(inv_sqrt2 * x[i]) + x[i] * coef * exp(-0.5 * x[i] * x[i])) * dy[i];
  }
}

__global__ void NaiveHalfGeluForwardGpu(const int64_t n, const half* x, const float inv_sqrt2,
                                        half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    float f_x = __half2float(x[i]);
    y[i] = __float2half(fgelu_forward(f_x, inv_sqrt2));
  }
}

__global__ void NaiveHalfGeluBackwardGpu(const int64_t n, const half* x, const half* dy,
                                         const float inv_sqrt2, const float coef, half* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    float f_x = __half2float(x[i]);
    float f_dy = __half2float(dy[i]);
    dx[i] = __float2half(fgelu_backward(f_x, f_dy, inv_sqrt2, coef));
  }
}

}  // namespace

template<typename T>
class GpuGeluKernel final : public user_op::OpKernel {
 public:
  GpuGeluKernel() = default;
  ~GpuGeluKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const T inv_sqrt2 = sqrt(0.5);
    RUN_CUDA_KERNEL((GeluForwardGpu<T>), ctx->device_ctx(), elem_cnt, elem_cnt, x->dptr<T>(),
                    inv_sqrt2, y->mut_dptr<T>());
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<>
class GpuGeluKernel<float16> final : public user_op::OpKernel {
 public:
  GpuGeluKernel() = default;
  ~GpuGeluKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const float inv_sqrt2 = sqrt(0.5);
    RUN_CUDA_KERNEL(NaiveHalfGeluForwardGpu, ctx->device_ctx(), elem_cnt, elem_cnt,
                    reinterpret_cast<const half*>(x->dptr<float16>()), inv_sqrt2,
                    reinterpret_cast<half*>(y->mut_dptr<float16>()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_GELU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("gelu").SetCreateFn<GpuGeluKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "gpu")                                            \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_GELU_KERNEL(float)
REGISTER_GPU_GELU_KERNEL(double)
REGISTER_GPU_GELU_KERNEL(float16)

template<typename T>
class GpuGeluGradKernel final : public user_op::OpKernel {
 public:
  GpuGeluGradKernel() = default;
  ~GpuGeluGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const T inv_sqrt2 = sqrt(0.5);
    const T coef = sqrt(2.0 / acos(-1.0));
    RUN_CUDA_KERNEL((GeluBackwardGpu<T>), ctx->device_ctx(), elem_cnt, elem_cnt, x->dptr<T>(),
                    dy->dptr<T>(), inv_sqrt2, coef, dx->mut_dptr<T>());
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<>
class GpuGeluGradKernel<float16> final : public user_op::OpKernel {
 public:
  GpuGeluGradKernel() = default;
  ~GpuGeluGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const float inv_sqrt2 = sqrt(0.5);
    const float coef = sqrt(2.0 / acos(-1.0));
    RUN_CUDA_KERNEL(NaiveHalfGeluBackwardGpu, ctx->device_ctx(), elem_cnt, elem_cnt,
                    reinterpret_cast<const half*>(x->dptr<float16>()),
                    reinterpret_cast<const half*>(dy->dptr<float16>()), inv_sqrt2, coef,
                    reinterpret_cast<half*>(dx->mut_dptr<float16>()));
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_GELU_GRAD_KERNEL(dtype)              \
  REGISTER_USER_KERNEL("gelu_grad")                       \
      .SetCreateFn<GpuGeluGradKernel<dtype>>()            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_GPU_GELU_GRAD_KERNEL(float)
REGISTER_GPU_GELU_GRAD_KERNEL(double)
REGISTER_GPU_GELU_GRAD_KERNEL(float16)

}  // namespace oneflow
