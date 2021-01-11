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
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace user_op {
template<template<typename> class Functor, typename T>
class CpuElementwiseMaximumMinimumKernel final : public user_op::OpKernel {
 public:
  CpuElementwiseMaximumMinimumKernel() = default;
  ~CpuElementwiseMaximumMinimumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
    const T* x = tensor_x->dptr<T>();
    const T* y = tensor_y->dptr<T>();
    T* z = tensor_z->mut_dptr<T>();
    int64_t n = tensor_x->shape().elem_cnt();
    FOR_RANGE(int64_t, i, 0, n) { z[i] = Functor<T>::Forward(x[i], y[i]); }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename> class Functor, typename T>
class CpuElementwiseMaximumMinimumBackwardKernel final : public user_op::OpKernel {
 public:
  CpuElementwiseMaximumMinimumBackwardKernel() = default;
  ~CpuElementwiseMaximumMinimumBackwardKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);

    const T* dptr_dz = tensor_dz->dptr<T>();
    const T* dptr_x = tensor_x->dptr<T>();
    const T* dptr_y = tensor_y->dptr<T>();

    T* dptr_dx = tensor_dx ? tensor_dx->mut_dptr<T>() : nullptr;
    T* dptr_dy = tensor_dy ? tensor_dy->mut_dptr<T>() : nullptr;
    size_t bytes_size = tensor_dz->shape().elem_cnt() * GetSizeOfDataType(tensor_dz->data_type());
    if (dptr_x) { Memset<DeviceType::kCPU>(ctx->device_ctx(), dptr_dx, 0, bytes_size); }
    if (dptr_y) { Memset<DeviceType::kCPU>(ctx->device_ctx(), dptr_dy, 0, bytes_size); }

    Functor<T>::Backward(tensor_dz->shape().elem_cnt(), dptr_dz, dptr_x, dptr_y, dptr_dx, dptr_dy);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FORWARD_CPU_KERNEL(op_type_name, functor, dtype)                    \
  REGISTER_USER_KERNEL(op_type_name)                                                 \
      .SetCreateFn<CpuElementwiseMaximumMinimumKernel<functor, dtype>>()             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU)                 \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#define REGISTER_BACKWARD_CPU_KERNEL(op_type_name, functor, dtype)               \
  REGISTER_USER_KERNEL(std::string("") + op_type_name + "_backward")             \
      .SetCreateFn<CpuElementwiseMaximumMinimumBackwardKernel<functor, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU)             \
                       & (user_op::HobDataType("dz", 0) == GetDataType<dtype>::value));

#define REGISTER_FORWARD_BACKWARD_KERNELS(op_type_name, functor, dtype) \
  REGISTER_FORWARD_CPU_KERNEL(op_type_name, functor, dtype);            \
  REGISTER_BACKWARD_CPU_KERNEL(op_type_name, functor, dtype);

namespace {
template<typename T>
struct CpuMaximumFunctor {
  static T Forward(const T x, const T y) { return x > y ? x : y; }

  static void Backward(const int64_t n, const T* dz, const T* x, const T* y, T* dx, T* dy) {
    FOR_RANGE(int64_t, idx, 0, n) {
      if (x[idx] > y[idx]) {
        if (dx) { dx[idx] = dz[idx]; }
      } else {
        if (dy) { dy[idx] = dz[idx]; }
      }
    }
  }
};

template<typename T>
struct CpuMinimumFunctor {
  static T Forward(const T x, const T y) { return x < y ? x : y; }

  static void Backward(const int64_t n, const T* dz, const T* x, const T* y, T* dx, T* dy) {
    FOR_RANGE(int64_t, idx, 0, n) {
      if (x[idx] < y[idx]) {
        if (dx) { dx[idx] = dz[idx]; }
      } else {
        if (dy) { dy[idx] = dz[idx]; }
      }
    }
  }
};
}  // namespace

REGISTER_FORWARD_BACKWARD_KERNELS("elementwise_maximum", CpuMaximumFunctor, float);
REGISTER_FORWARD_BACKWARD_KERNELS("elementwise_maximum", CpuMaximumFunctor, double);
REGISTER_FORWARD_BACKWARD_KERNELS("elementwise_minimum", CpuMinimumFunctor, float);
REGISTER_FORWARD_BACKWARD_KERNELS("elementwise_minimum", CpuMinimumFunctor, double);
}  // namespace user_op
}  // namespace oneflow
