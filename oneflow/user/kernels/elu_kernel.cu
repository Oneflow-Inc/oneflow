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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

namespace user_op {

template<typename T>
struct EluFunctor {
  __host__ __device__ explicit EluFunctor(T alpha) : alpha(alpha) {}
  __device__ T operator()(T x) const {
    return (x > static_cast<T>(0)) ? x : alpha * (exp(x) - static_cast<T>(1));
  }
  const T alpha;
};

template<typename T>
struct EluGradFunctor {
  __host__ __device__ explicit EluGradFunctor(T alpha) : alpha(alpha) {}
  __device__ T operator()(T x, T dy) const {
    return (x > static_cast<T>(0)) ? dy : dy * alpha * (exp(x));
  }
  const T alpha;
};

template<>
struct EluFunctor<half> {
  __host__ __device__ explicit EluFunctor(float alpha)
      : alpha(alpha), float_functor(EluFunctor<float>(alpha)) {}
  __device__ half operator()(half x) const { return __float2half(float_functor(__half2float(x))); }
  const float alpha;
  EluFunctor<float> float_functor;
};

template<>
struct EluGradFunctor<half> {
  __host__ __device__ explicit EluGradFunctor(float alpha)
      : alpha(alpha), float_functor(EluGradFunctor<float>(alpha)) {}
  __device__ half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
  const float alpha;
  EluGradFunctor<float> float_functor;
};

template<DeviceType device_type, typename T>
class GpuEluKernel final : public OpKernel {
 public:
  GpuEluKernel() = default;
  ~GpuEluKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const T alpha = static_cast<T>(ctx->Attr<double>("alpha"));
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();

    const int32_t elem_cnt = in_tensor->shape().elem_cnt();
    OF_CUDA_CHECK((oneflow::cuda::elementwise::Unary(EluFunctor<T>(alpha), elem_cnt, out_ptr,
                                                     in_ptr, ctx->device_ctx()->cuda_stream())));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_ELU_KERNEL(device, dtype)                                            \
  REGISTER_USER_KERNEL("elu").SetCreateFn<GpuEluKernel<device, dtype>>().SetIsMatchedHob( \
      (HobDeviceTag() == device) & (HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_ELU_KERNEL(DeviceType::kGPU, half);
REGISTER_GPU_ELU_KERNEL(DeviceType::kGPU, float);
REGISTER_GPU_ELU_KERNEL(DeviceType::kGPU, double);

template<DeviceType device_type, typename T>
class GpuEluGradKernel final : public OpKernel {
 public:
  GpuEluGradKernel() = default;
  ~GpuEluGradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const T alpha = static_cast<T>(ctx->Attr<double>("alpha"));
    const T* x_ptr = x_tensor->dptr<T>();
    const T* dy_ptr = dy_tensor->dptr<T>();
    T* dx_ptr = dx_tensor->mut_dptr<T>();
    const int32_t elem_cnt = x_tensor->shape().elem_cnt();
    OF_CUDA_CHECK(
        (oneflow::cuda::elementwise::Binary(EluGradFunctor<T>(alpha), elem_cnt, dx_ptr, x_ptr,
                                            dy_ptr, ctx->device_ctx()->cuda_stream())));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_ELU_BACKWARD_KERNEL(device, dtype) \
  REGISTER_USER_KERNEL("elu_grad")                      \
      .SetCreateFn<GpuEluGradKernel<device, dtype>>()   \
      .SetIsMatchedHob((HobDeviceTag() == device)       \
                       & (HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_GPU_ELU_BACKWARD_KERNEL(DeviceType::kGPU, half);
REGISTER_GPU_ELU_BACKWARD_KERNEL(DeviceType::kGPU, float);
REGISTER_GPU_ELU_BACKWARD_KERNEL(DeviceType::kGPU, double);

}  // namespace user_op

}  // namespace oneflow
