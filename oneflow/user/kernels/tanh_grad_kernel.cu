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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace user_op {

namespace {

template<typename T>
struct TanhGradFunctor;

template<>
struct TanhGradFunctor<float> {
  OF_DEVICE_FUNC float operator()(float x, float dy) const {
    float tanh_val = tanhf(x);
    return dy * (static_cast<float>(1.0) - tanh_val * tanh_val);
  }
};

template<>
struct TanhGradFunctor<double> {
  OF_DEVICE_FUNC double operator()(double x, double dy) const {
    double tanh_val = tanh(x);
    return dy * (static_cast<double>(1.0) - tanh_val * tanh_val);
  }
};

template<>
struct TanhGradFunctor<half> {
  TanhGradFunctor<float> float_functor;
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
};

#if CUDA_VERSION >= 11000
template<>
struct TanhGradFunctor<nv_bfloat16> {
  TanhGradFunctor<float> float_functor;
  OF_DEVICE_FUNC nv_bfloat16 operator()(nv_bfloat16 x, nv_bfloat16 dy) const {
    return __float2bfloat16(float_functor(__bfloat162float(x), __bfloat162float(dy)));
  }
};
#endif

}  // namespace

template<typename T>
class TanhGradGPUKernel final : public OpKernel {
 public:
  TanhGradGPUKernel() = default;
  ~TanhGradGPUKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t elem_cnt = x->shape_view().elem_cnt();
    const T* x_ptr = reinterpret_cast<const T*>(x->dptr());
    const T* dy_ptr = reinterpret_cast<const T*>(dy->dptr());
    T* dx_ptr = reinterpret_cast<T*>(dx->mut_dptr());
    OF_CUDA_CHECK(cuda::elementwise::Binary(TanhGradFunctor<T>(), elem_cnt, dx_ptr, x_ptr, dy_ptr,
                                            ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_TANH_GRAD_KERNEL_GPU(cpp_type, data_type)             \
  REGISTER_USER_KERNEL((std::string("") + "tanh" + "_grad"))           \
      .SetCreateFn<TanhGradGPUKernel<cpp_type>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dx", 0) == data_type))

REGISTER_TANH_GRAD_KERNEL_GPU(half, DataType::kFloat16);
REGISTER_TANH_GRAD_KERNEL_GPU(float, DataType::kFloat);
REGISTER_TANH_GRAD_KERNEL_GPU(double, DataType::kDouble);
#if CUDA_VERSION >= 11000
REGISTER_TANH_GRAD_KERNEL_GPU(nv_bfloat16, DataType::kBFloat16);
#endif

}  // namespace user_op

}  // namespace oneflow
