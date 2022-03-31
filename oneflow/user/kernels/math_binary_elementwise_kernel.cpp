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
#include "oneflow/user/kernels/math_binary_elementwise_func.h"
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/ep/cpu/cpu_device.h"

namespace oneflow {

template<template<typename> class BinaryFunctor, typename T>
class MathBinaryElementwiseCpuKernel final : public user_op::OpKernel {
 public:
  MathBinaryElementwiseCpuKernel() = default;
  ~MathBinaryElementwiseCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
    const T* x = tensor_x->dptr<T>();
    const T* y = tensor_y->dptr<T>();
    T* z = tensor_z->mut_dptr<T>();
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    ep::CpuStream* cpu_stream = ctx->stream()->As<ep::CpuStream>();

    bool x_contiguous = oneflow::one::IsContiguous(tensor_x);
    bool y_contiguous = oneflow::one::IsContiguous(tensor_y);
    if (x_contiguous && y_contiguous) {
      cpu_stream->ParallelFor(0, n, [x, y, z](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) { z[i] = BinaryFunctor<T>::Forward(x[i], y[i]); }
      });
    } else if (x_contiguous) {
      StrideParam y_stride = oneflow::one::GetStrideParam(tensor_y);
      StrideParam z_stride = oneflow::one::GetStrideParam(tensor_z);
      cpu_stream->ParallelFor(0, n, [x, y, z, y_stride, z_stride](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          int32_t y_idx = compute_index(i, y_stride, z_stride);
          z[i] = BinaryFunctor<T>::Forward(x[i], y[y_idx]);
        }
      });
    } else if (y_contiguous) {
      StrideParam x_stride = oneflow::one::GetStrideParam(tensor_x);
      StrideParam z_stride = oneflow::one::GetStrideParam(tensor_z);
      cpu_stream->ParallelFor(0, n, [x, y, z, x_stride, z_stride](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          int32_t x_idx = compute_index(i, x_stride, z_stride);
          z[i] = BinaryFunctor<T>::Forward(x[x_idx], y[i]);
        }
      });
    } else {
      StrideParam x_stride = oneflow::one::GetStrideParam(tensor_x);
      StrideParam y_stride = oneflow::one::GetStrideParam(tensor_y);
      StrideParam z_stride = oneflow::one::GetStrideParam(tensor_z);
      cpu_stream->ParallelFor(0, n,
                              [x, y, z, x_stride, y_stride, z_stride](int64_t begin, int64_t end) {
                                for (int64_t i = begin; i < end; i++) {
                                  int32_t x_idx = compute_index(i, x_stride, z_stride);
                                  int32_t y_idx = compute_index(i, y_stride, z_stride);
                                  z[i] = BinaryFunctor<T>::Forward(x[x_idx], y[y_idx]);
                                }
                              });
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename> class BinaryFunctor, typename T>
class MathBinaryElementwiseXGradCpuKernel final : public user_op::OpKernel {
 public:
  MathBinaryElementwiseXGradCpuKernel() = default;
  ~MathBinaryElementwiseXGradCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const T* x = tensor_x->dptr<T>();
    const T* y = tensor_y->dptr<T>();
    const T* dz = tensor_dz->dptr<T>();
    T* dx = tensor_dx->mut_dptr<T>();
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);

    const bool x_contiguous = oneflow::one::IsContiguous(tensor_x);
    const bool y_contiguous = oneflow::one::IsContiguous(tensor_y);
    const bool dz_contiguous = oneflow::one::IsContiguous(tensor_dz);
    if (x_contiguous && y_contiguous && dz_contiguous) {
      for (int32_t i = 0; i < n; ++i) {
        dx[i] = BinaryFunctor<T>::BackwardXGrad(x[i], y[i], dz[i]);
      }
    } else if (x_contiguous && y_contiguous && !dz_contiguous) {
      const StrideParam dz_stride = oneflow::one::GetStrideParam(tensor_dz);
      const StrideParam dx_stride = oneflow::one::GetStrideParam(tensor_dx);
      for (int32_t i = 0; i < n; ++i) {
        const int32_t dz_idx = compute_index(i, dz_stride, dx_stride);
        dx[i] = BinaryFunctor<T>::BackwardXGrad(x[i], y[i], dz[dz_idx]);
      }
    } else if (x_contiguous && !y_contiguous && dz_contiguous) {
      const StrideParam y_stride = oneflow::one::GetStrideParam(tensor_y);
      const StrideParam dx_stride = oneflow::one::GetStrideParam(tensor_dx);
      for (int32_t i = 0; i < n; ++i) {
        const int32_t y_idx = compute_index(i, y_stride, dx_stride);
        dx[i] = BinaryFunctor<T>::BackwardXGrad(x[i], y[y_idx], dz[i]);
      }
    } else if (!x_contiguous && y_contiguous && dz_contiguous) {
      const StrideParam x_stride = oneflow::one::GetStrideParam(tensor_x);
      const StrideParam dx_stride = oneflow::one::GetStrideParam(tensor_dx);
      for (int32_t i = 0; i < n; ++i) {
        const int32_t x_idx = compute_index(i, x_stride, dx_stride);
        dx[i] = BinaryFunctor<T>::BackwardXGrad(x[x_idx], y[i], dz[i]);
      }
    } else if (!x_contiguous && !y_contiguous && dz_contiguous) {
      const StrideParam x_stride = oneflow::one::GetStrideParam(tensor_x);
      const StrideParam y_stride = oneflow::one::GetStrideParam(tensor_y);
      const StrideParam dx_stride = oneflow::one::GetStrideParam(tensor_dx);
      for (int32_t i = 0; i < n; ++i) {
        const int32_t x_idx = compute_index(i, x_stride, dx_stride);
        const int32_t y_idx = compute_index(i, y_stride, dx_stride);
        dx[i] = BinaryFunctor<T>::BackwardXGrad(x[x_idx], y[y_idx], dz[i]);
      }
    } else if (!x_contiguous && y_contiguous && !dz_contiguous) {
      const StrideParam x_stride = oneflow::one::GetStrideParam(tensor_x);
      const StrideParam dz_stride = oneflow::one::GetStrideParam(tensor_dz);
      const StrideParam dx_stride = oneflow::one::GetStrideParam(tensor_dx);
      for (int32_t i = 0; i < n; ++i) {
        const int32_t x_idx = compute_index(i, x_stride, dx_stride);
        const int32_t dz_idx = compute_index(i, dz_stride, dx_stride);
        dx[i] = BinaryFunctor<T>::BackwardXGrad(x[x_idx], y[i], dz[dz_idx]);
      }
    } else if (x_contiguous && !y_contiguous && !dz_contiguous) {
      const StrideParam y_stride = oneflow::one::GetStrideParam(tensor_y);
      const StrideParam dz_stride = oneflow::one::GetStrideParam(tensor_dz);
      const StrideParam dx_stride = oneflow::one::GetStrideParam(tensor_dx);
      for (int32_t i = 0; i < n; ++i) {
        const int32_t y_idx = compute_index(i, y_stride, dx_stride);
        const int32_t dz_idx = compute_index(i, dz_stride, dx_stride);
        dx[i] = BinaryFunctor<T>::BackwardXGrad(x[i], y[y_idx], dz[dz_idx]);
      }
    } else if (!x_contiguous && !y_contiguous && !dz_contiguous) {
      const StrideParam x_stride = oneflow::one::GetStrideParam(tensor_x);
      const StrideParam y_stride = oneflow::one::GetStrideParam(tensor_y);
      const StrideParam dz_stride = oneflow::one::GetStrideParam(tensor_dz);
      const StrideParam dx_stride = oneflow::one::GetStrideParam(tensor_dx);
      for (int32_t i = 0; i < n; ++i) {
        const int32_t x_idx = compute_index(i, x_stride, dx_stride);
        const int32_t y_idx = compute_index(i, y_stride, dx_stride);
        const int32_t dz_idx = compute_index(i, dz_stride, dx_stride);
        dx[i] = BinaryFunctor<T>::BackwardXGrad(x[x_idx], y[y_idx], dz[dz_idx]);
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename> class BinaryFunctor, typename T>
class MathBinaryElementwiseYGradCpuKernel final : public user_op::OpKernel {
 public:
  MathBinaryElementwiseYGradCpuKernel() = default;
  ~MathBinaryElementwiseYGradCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);

    const T* x = tensor_x->dptr<T>();
    const T* y = tensor_y->dptr<T>();
    const T* dz = tensor_dz->dptr<T>();
    T* dy = tensor_dy->mut_dptr<T>();
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    for (int32_t i = 0; i < n; ++i) { dy[i] = BinaryFunctor<T>::BackwardYGrad(x[i], y[i], dz[i]); }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MATH_BINARY_ELEMENTWISE_CPU_KERNEL_AND_GRAD(math_type_pair, data_type_pair)    \
  REGISTER_USER_KERNEL(OF_PP_PAIR_FIRST(math_type_pair))                                        \
      .SetCreateFn<                                                                             \
          MathBinaryElementwiseCpuKernel<OF_PP_CAT(OF_PP_PAIR_SECOND(math_type_pair), Functor), \
                                         OF_PP_PAIR_FIRST(data_type_pair)>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                           \
                       && (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(data_type_pair))); \
                                                                                                \
  REGISTER_USER_KERNEL((std::string("") + OF_PP_PAIR_FIRST(math_type_pair) + "_x_grad"))        \
      .SetCreateFn<MathBinaryElementwiseXGradCpuKernel<                                         \
          OF_PP_CAT(OF_PP_PAIR_SECOND(math_type_pair), Functor),                                \
          OF_PP_PAIR_FIRST(data_type_pair)>>()                                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                           \
                       && (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(data_type_pair))); \
  REGISTER_USER_KERNEL((std::string("") + OF_PP_PAIR_FIRST(math_type_pair) + "_y_grad"))        \
      .SetCreateFn<MathBinaryElementwiseYGradCpuKernel<                                         \
          OF_PP_CAT(OF_PP_PAIR_SECOND(math_type_pair), Functor),                                \
          OF_PP_PAIR_FIRST(data_type_pair)>>()                                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                           \
                       && (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(data_type_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_ELEMENTWISE_CPU_KERNEL_AND_GRAD,
                                 MATH_BINARY_ELEMENTWISE_FUNC_SEQ, FLOATING_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_ELEMENTWISE_CPU_KERNEL_AND_GRAD,
                                 OF_PP_MAKE_TUPLE_SEQ("floordiv", FloorDiv),
                                 INT_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ)

}  // namespace oneflow
