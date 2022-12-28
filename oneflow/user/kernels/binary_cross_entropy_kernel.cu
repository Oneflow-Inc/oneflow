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
#include "oneflow/user/kernels/loss_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace user_op {
namespace {

using namespace loss;

template<typename T>
struct BinaryCrossEntropyFunctor {
  T zero_;
  T one_;
  T negative_hundred_;
  BinaryCrossEntropyFunctor()
      : zero_(GetZeroVal<T>()), one_(GetOneVal<T>()), negative_hundred_(static_cast<T>(-100)) {}
  __device__ __forceinline__ T operator()(T input_val, T target_val) const {
    assert(input_val >= zero_);
    assert(input_val <= one_);
    return (target_val - one_) * max(static_cast<T>(log(one_ - input_val)), negative_hundred_)
           - target_val * max(static_cast<T>(log(input_val)), negative_hundred_);
  }

  __device__ __forceinline__ T operator()(T input_val, T target_val, T weight_val) const {
    return (*this)(input_val, target_val) * weight_val;
  }
};

template<>
struct BinaryCrossEntropyFunctor<float> {
  float zero_;
  float one_;
  float negative_hundred_;
  BinaryCrossEntropyFunctor() : zero_(0.f), one_(1.f), negative_hundred_(-100.f) {}
  __device__ __forceinline__ float operator()(float input_val, float target_val) const {
    assert(input_val >= zero_);
    assert(input_val <= one_);
    return (target_val - one_) * max(logf(one_ - input_val), negative_hundred_)
           - target_val * max(logf(input_val), negative_hundred_);
  }

  __device__ __forceinline__ float operator()(float input_val, float target_val,
                                              float weight_val) const {
    return (*this)(input_val, target_val) * weight_val;
  }
};

template<>
struct BinaryCrossEntropyFunctor<half> {
  BinaryCrossEntropyFunctor<float> float_functor;
  __device__ __forceinline__ half operator()(half input_val, half target_val) const {
    return __float2half(float_functor(__half2float(input_val), __half2float(target_val)));
  }

  __device__ __forceinline__ half operator()(half input_val, half target_val,
                                             half weight_val) const {
    return (*this)(input_val, target_val) * weight_val;
  }
};

template<typename T>
struct BinaryCrossEntropyGradFunctor {
  T eps_;
  T one_;
  BinaryCrossEntropyGradFunctor() : eps_(static_cast<T>(1e-12)), one_(GetOneVal<T>()) {}
  __device__ __forceinline__ T operator()(T input_val, T target_val, T dy_val) const {
    return dy_val * (input_val - target_val) / max((one_ - input_val) * input_val, eps_);
  }
  __device__ __forceinline__ T operator()(T input_val, T target_val, T dy_val, T weight_val) const {
    return (*this)(input_val, target_val, dy_val) * weight_val;
  }
};

template<>
struct BinaryCrossEntropyGradFunctor<half> {
  BinaryCrossEntropyGradFunctor<float> float_functor;
  BinaryCrossEntropyGradFunctor() {}
  __device__ __forceinline__ half operator()(half input_val, half target_val, half dy_val) const {
    return __float2half(
        float_functor(__half2float(input_val), __half2float(target_val), __half2float(dy_val)));
  }
  __device__ __forceinline__ half operator()(half input_val, half target_val, half dy_val,
                                             half weight_val) const {
    return __float2half(float_functor(__half2float(input_val), __half2float(target_val),
                                      __half2float(dy_val), __half2float(weight_val)));
  }
};

template<typename T>
class BinaryCrossEntropyKernel final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyKernel() = default;
  ~BinaryCrossEntropyKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int64_t elem_cnt = input_blob->shape_view().elem_cnt();

    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* out = out_blob->mut_dptr<T>();

    if (ctx->has_input("weight", 0)) {
      const T* weight = ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>();
      OF_CUDA_CHECK(
          (cuda::elementwise::Ternary(BinaryCrossEntropyFunctor<T>(), elem_cnt, out, input, target,
                                      weight, ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
    } else {
      OF_CUDA_CHECK(
          (cuda::elementwise::Binary(BinaryCrossEntropyFunctor<T>(), elem_cnt, out, input, target,
                                     ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class BinaryCrossEntropyGradKernel final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyGradKernel() = default;
  ~BinaryCrossEntropyGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    const auto* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const int64_t elem_cnt = input_blob->shape_view().elem_cnt();

    const T* dy = dy_blob->dptr<T>();
    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* dx = dx_blob->mut_dptr<T>();

    if (ctx->has_input("weight", 0)) {
      const T* weight = ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>();
      using FunctorT = BinaryCrossEntropyGradFunctor<T>;
      using FactoryT = cuda::elementwise::SimpleFactory<FunctorT>;
      OF_CUDA_CHECK((cuda::elementwise::GenericLauncher<FactoryT, T, T, T, T, T>::Launch(
          FactoryT(FunctorT()), elem_cnt, dx, input, target, dy, weight,
          ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
    } else {
      OF_CUDA_CHECK((cuda::elementwise::Ternary(
          BinaryCrossEntropyGradFunctor<T>(), elem_cnt, dx, input, target, dy,
          ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_BINARY_CROSS_ENTROPY_KERNEL(dtype)                                        \
  REGISTER_USER_KERNEL("binary_cross_entropy")                                             \
      .SetCreateFn<BinaryCrossEntropyKernel<dtype>>()                                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       && (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

#define REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("binary_cross_entropy_grad")                                        \
      .SetCreateFn<BinaryCrossEntropyGradKernel<dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       && (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_BINARY_CROSS_ENTROPY_KERNEL(half)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(double)

REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(half)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(double)

}  // namespace user_op
}  // namespace oneflow
