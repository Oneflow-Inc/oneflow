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
#ifdef WITH_CUDA
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {
namespace user_op {

namespace {
template<typename T>
struct GpuMaximumFunctor {
  OF_DEVICE_FUNC T operator()(T x, T y) const { return x > y ? x : y; }

  OF_DEVICE_FUNC static void Backward(int64_t elem_cnt, const T* dz, const T* x, const T* y, T* dx,
                                      T* dy) {
    XPU_1D_KERNEL_LOOP(idx, elem_cnt) {
      if (x[idx] > y[idx]) {
        if (dx) { dx[idx] = dz[idx]; }
      } else {
        if (dy) { dy[idx] = dz[idx]; }
      }
    }
  }
};

template<typename T>
struct GpuMinimumFunctor {
  OF_DEVICE_FUNC T operator()(T x, T y) const { return x < y ? x : y; }

  OF_DEVICE_FUNC static void Backward(int64_t elem_cnt, const T* dz, const T* x, const T* y, T* dx,
                                      T* dy) {
    XPU_1D_KERNEL_LOOP(idx, elem_cnt) {
      if (x[idx] < y[idx]) {
        if (dx) { dx[idx] = dz[idx]; }
      } else {
        if (dy) { dy[idx] = dz[idx]; }
      }
    }
  }
};
}  // namespace

template<template<typename> class BackwardFunctor, typename T>
__global__ void ElementwiseBackwardGradGpu(int64_t elem_cnt, const T* dz, const T* x, const T* y,
                                           T* dx, T* dy) {
  BackwardFunctor<T>::Backward(elem_cnt, dz, x, y, dx, dy);
}

template<template<typename> class Functor, typename T>
class GpuElementwiseMaximumMinimumKernel final : public user_op::OpKernel {
 public:
  GpuElementwiseMaximumMinimumKernel() = default;
  ~GpuElementwiseMaximumMinimumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
    int64_t n = tensor_x->shape().elem_cnt();

    OF_CUDA_CHECK(cuda::elementwise::Binary(Functor<T>(), n, tensor_z->mut_dptr<T>(),
                                            tensor_x->dptr<T>(), tensor_y->dptr<T>(),
                                            ctx->device_ctx()->cuda_stream()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename> class GradFunctor, typename T>
class GpuElementwiseMaximumMinimumBackwardKernel final : public user_op::OpKernel {
 public:
  GpuElementwiseMaximumMinimumBackwardKernel() = default;
  ~GpuElementwiseMaximumMinimumBackwardKernel() = default;

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

    const int cnt = tensor_dz->shape().elem_cnt();
    size_t bytes_size = cnt * GetSizeOfDataType(tensor_dz->data_type());
    if (dptr_x) { Memset<DeviceType::kGPU>(ctx->device_ctx(), dptr_dx, 0, bytes_size); }
    if (dptr_y) { Memset<DeviceType::kGPU>(ctx->device_ctx(), dptr_dy, 0, bytes_size); }

    ElementwiseBackwardGradGpu<GradFunctor, T>
        <<<BlocksNum4ThreadsNum(cnt), kCudaThreadsNumPerBlock, 0,
           ctx->device_ctx()->cuda_stream()>>>(cnt, dptr_dz, dptr_x, dptr_y, dptr_dx, dptr_dy);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FORWARD_GPU_KERNEL(op_type_name, functor, dtype)                    \
  REGISTER_USER_KERNEL(op_type_name)                                                 \
      .SetCreateFn<GpuElementwiseMaximumMinimumKernel<functor, dtype>>()             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                 \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#define REGISTER_BACKWARD_GPU_KERNEL(op_type_name, functor, dtype)               \
  REGISTER_USER_KERNEL(std::string("") + op_type_name + "_backward")             \
      .SetCreateFn<GpuElementwiseMaximumMinimumBackwardKernel<functor, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)             \
                       & (user_op::HobDataType("dz", 0) == GetDataType<dtype>::value));

#define REGISTER_FORWARD_BACKWARD_GPU_KERNELS(op_type_name, functor, dtype) \
  REGISTER_FORWARD_GPU_KERNEL(op_type_name, functor, dtype);                \
  REGISTER_BACKWARD_GPU_KERNEL(op_type_name, functor, dtype);

REGISTER_FORWARD_BACKWARD_GPU_KERNELS("elementwise_maximum", GpuMaximumFunctor, float);
REGISTER_FORWARD_BACKWARD_GPU_KERNELS("elementwise_maximum", GpuMaximumFunctor, double);
REGISTER_FORWARD_BACKWARD_GPU_KERNELS("elementwise_minimum", GpuMinimumFunctor, float);
REGISTER_FORWARD_BACKWARD_GPU_KERNELS("elementwise_minimum", GpuMinimumFunctor, double);

}  // namespace user_op
}  // namespace oneflow
#endif  // WITH_CUDA
