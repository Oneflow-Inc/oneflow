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
#include "oneflow/core/cuda/elementwise.cuh"

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

}  // namespace

template<typename T>
class GpuGeluKernel final : public user_op::OpKernel {
 public:
  GpuGeluKernel() = default;
  ~GpuGeluKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t elem_cnt = x->shape().elem_cnt();
    OF_CUDA_CHECK((cuda::elementwise::Unary(GeluFunctor<T>(), elem_cnt, y->mut_dptr<T>(),
                                            x->dptr<T>(), ctx->device_ctx()->cuda_stream())));
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_GELU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("gelu").SetCreateFn<GpuGeluKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "gpu")                                            \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_GELU_KERNEL(float)
REGISTER_GPU_GELU_KERNEL(double)
REGISTER_GPU_GELU_KERNEL(half)

template<typename T>
class GpuGeluGradKernel final : public user_op::OpKernel {
 public:
  GpuGeluGradKernel() = default;
  ~GpuGeluGradKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t elem_cnt = x->shape().elem_cnt();
    OF_CUDA_CHECK(
        (cuda::elementwise::Binary(GeluGradFunctor<T>(), elem_cnt, dx->mut_dptr<T>(), x->dptr<T>(),
                                   dy->dptr<T>(), ctx->device_ctx()->cuda_stream())));
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
REGISTER_GPU_GELU_GRAD_KERNEL(half)

}  // namespace oneflow
