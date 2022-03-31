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
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

namespace {

template<template<typename> class BinaryFunctor, typename T>
__global__ void MathBinaryElementwiseForwardGpu(const int n, const T* x, const T* y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = BinaryFunctor<T>::Forward(x[i], y[i]); }
}

template<template<typename> class BinaryFunctor, typename T>
__global__ void MathBinaryElementwiseWithXStrideForwardGpu(const int n, const StrideParam x_stride,
                                                           const StrideParam z_stride, const T* x,
                                                           const T* y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const int32_t x_idx = oneflow::cuda::elementwise::offset_to_index(i, x_stride, z_stride);
    z[i] = BinaryFunctor<T>::Forward(x[x_idx], y[i]);
  }
}

template<template<typename> class BinaryFunctor, typename T>
__global__ void MathBinaryElementwiseWithYStrideForwardGpu(const int n, const StrideParam y_stride,
                                                           const StrideParam z_stride, const T* x,
                                                           const T* y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const int32_t y_idx = oneflow::cuda::elementwise::offset_to_index(i, y_stride, z_stride);
    z[i] = BinaryFunctor<T>::Forward(x[i], y[y_idx]);
  }
}

template<template<typename> class BinaryFunctor, typename T>
__global__ void MathBinaryElementwiseWithStrideForwardGpu(const int n, const StrideParam x_stride,
                                                          const StrideParam y_stride,
                                                          const StrideParam z_stride, const T* x,
                                                          const T* y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const int32_t x_idx = oneflow::cuda::elementwise::offset_to_index(i, x_stride, z_stride);
    const int32_t y_idx = oneflow::cuda::elementwise::offset_to_index(i, y_stride, z_stride);
    z[i] = BinaryFunctor<T>::Forward(x[x_idx], y[y_idx]);
  }
}

template<template<typename> class BinaryFunctor, typename T>
__global__ void MathBinaryElementwiseBackwardXGradGpu(const int n, const T* x, const T* y,
                                                      const T* dz, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = BinaryFunctor<T>::BackwardXGrad(x[i], y[i], dz[i]); }
}

template<template<typename> class BinaryFunctor, typename T>
__global__ void MathBinaryElementwiseStrideBackwardXGradGpu(
    const int n, const T* x, const T* y, const T* dz, T* dx, const StrideParam& x_stride,
    const StrideParam& y_stride, const StrideParam& dz_stride, const StrideParam& dx_stride) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const int32_t x_idx = oneflow::cuda::elementwise::offset_to_index(i, x_stride, dx_stride);
    const int32_t y_idx = oneflow::cuda::elementwise::offset_to_index(i, y_stride, dx_stride);
    const int32_t dz_idx = oneflow::cuda::elementwise::offset_to_index(i, dz_stride, dx_stride);
    dx[i] = BinaryFunctor<T>::BackwardXGrad(x[x_idx], y[y_idx], dz[dz_idx]);
  }
}

template<template<typename> class BinaryFunctor, typename T>
__global__ void MathBinaryElementwiseBackwardYGradGpu(const int n, const T* x, const T* y,
                                                      const T* dz, T* dy) {
  CUDA_1D_KERNEL_LOOP(i, n) { dy[i] = BinaryFunctor<T>::BackwardYGrad(x[i], y[i], dz[i]); }
}

template<template<typename> class BinaryFunctor, typename T>
__global__ void MathBinaryElementwiseStrideBackwardYGradGpu(
    const int n, const T* x, const T* y, const T* dz, T* dy, const StrideParam& x_stride,
    const StrideParam& y_stride, const StrideParam& dz_stride, const StrideParam& dy_stride) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const int32_t x_idx = oneflow::cuda::elementwise::offset_to_index(i, x_stride, dy_stride);
    const int32_t y_idx = oneflow::cuda::elementwise::offset_to_index(i, y_stride, dy_stride);
    const int32_t dz_idx = oneflow::cuda::elementwise::offset_to_index(i, dz_stride, dy_stride);
    dy[i] = BinaryFunctor<T>::BackwardXGrad(x[x_idx], y[y_idx], dz[dz_idx]);
  }
}

}  // namespace

template<template<typename> class BinaryFunctor, typename T>
class MathBinaryElementwiseGpuKernel final : public user_op::OpKernel {
 public:
  MathBinaryElementwiseGpuKernel() = default;
  ~MathBinaryElementwiseGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    if (n == 0) { return; }

    // compute is_contiguous and construct input/output stride params
    bool x_contiguous = oneflow::one::IsContiguous(tensor_x);
    bool y_contiguous = oneflow::one::IsContiguous(tensor_y);
    StrideParam x_stride = oneflow::one::GetStrideParam(tensor_x);
    StrideParam y_stride = oneflow::one::GetStrideParam(tensor_y);
    StrideParam z_stride = oneflow::one::GetStrideParam(tensor_z);
    if (x_contiguous && y_contiguous) {
      MathBinaryElementwiseForwardGpu<BinaryFunctor, T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              n, tensor_x->dptr<T>(), tensor_y->dptr<T>(), tensor_z->mut_dptr<T>());
    } else if (x_contiguous) {
      MathBinaryElementwiseWithYStrideForwardGpu<BinaryFunctor, T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              n, y_stride, z_stride, tensor_x->dptr<T>(), tensor_y->dptr<T>(),
              tensor_z->mut_dptr<T>());
    } else if (y_contiguous) {
      MathBinaryElementwiseWithXStrideForwardGpu<BinaryFunctor, T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              n, x_stride, z_stride, tensor_x->dptr<T>(), tensor_y->dptr<T>(),
              tensor_z->mut_dptr<T>());
    } else {
      MathBinaryElementwiseWithStrideForwardGpu<BinaryFunctor, T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              n, x_stride, y_stride, z_stride, tensor_x->dptr<T>(), tensor_y->dptr<T>(),
              tensor_z->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename> class BinaryFunctor, typename T>
class MathBinaryElementwiseXGradGpuKernel final : public user_op::OpKernel {
 public:
  MathBinaryElementwiseXGradGpuKernel() = default;
  ~MathBinaryElementwiseXGradGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    if (n == 0) { return; }

    const bool x_contiguous = oneflow::one::IsContiguous(tensor_x);
    const bool y_contiguous = oneflow::one::IsContiguous(tensor_y);
    const bool dz_contiguous = oneflow::one::IsContiguous(tensor_dz);
    if (x_contiguous && y_contiguous && dz_contiguous) {
      MathBinaryElementwiseBackwardXGradGpu<BinaryFunctor, T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              n, tensor_x->dptr<T>(), tensor_y->dptr<T>(), tensor_dz->dptr<T>(),
              tensor_dx->mut_dptr<T>());
    } else {
      const StrideParam x_stride = oneflow::one::GetStrideParam(tensor_x);
      const StrideParam y_stride = oneflow::one::GetStrideParam(tensor_y);
      const StrideParam dz_stride = oneflow::one::GetStrideParam(tensor_dz);
      const StrideParam dx_stride = oneflow::one::GetStrideParam(tensor_dx);
      MathBinaryElementwiseStrideBackwardXGradGpu<BinaryFunctor, T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              n, tensor_x->dptr<T>(), tensor_y->dptr<T>(), tensor_dz->dptr<T>(),
              tensor_dx->mut_dptr<T>(), x_stride, y_stride, dz_stride, dx_stride);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename> class BinaryFunctor, typename T>
class MathBinaryElementwiseYGradGpuKernel final : public user_op::OpKernel {
 public:
  MathBinaryElementwiseYGradGpuKernel() = default;
  ~MathBinaryElementwiseYGradGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    if (n == 0) { return; }

    const bool x_contiguous = oneflow::one::IsContiguous(tensor_x);
    const bool y_contiguous = oneflow::one::IsContiguous(tensor_y);
    const bool dz_contiguous = oneflow::one::IsContiguous(tensor_dz);
    if (x_contiguous && y_contiguous && dz_contiguous) {
      MathBinaryElementwiseBackwardYGradGpu<BinaryFunctor, T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              n, tensor_x->dptr<T>(), tensor_y->dptr<T>(), tensor_dz->dptr<T>(),
              tensor_dy->mut_dptr<T>());
    } else {
      const StrideParam x_stride = oneflow::one::GetStrideParam(tensor_x);
      const StrideParam y_stride = oneflow::one::GetStrideParam(tensor_y);
      const StrideParam dz_stride = oneflow::one::GetStrideParam(tensor_dz);
      const StrideParam dy_stride = oneflow::one::GetStrideParam(tensor_dy);
      MathBinaryElementwiseStrideBackwardYGradGpu<BinaryFunctor, T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              n, tensor_x->dptr<T>(), tensor_y->dptr<T>(), tensor_dz->dptr<T>(),
              tensor_dy->mut_dptr<T>(), x_stride, y_stride, dz_stride, dy_stride);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MATH_BINARY_ELEMENTWISE_CUDA_KERNEL_AND_GRAD(math_type_pair, data_type_pair)   \
  REGISTER_USER_KERNEL(OF_PP_PAIR_FIRST(math_type_pair))                                        \
      .SetCreateFn<                                                                             \
          MathBinaryElementwiseGpuKernel<OF_PP_CAT(OF_PP_PAIR_SECOND(math_type_pair), Functor), \
                                         OF_PP_PAIR_FIRST(data_type_pair)>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                          \
                       && (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(data_type_pair))); \
                                                                                                \
  REGISTER_USER_KERNEL((std::string("") + OF_PP_PAIR_FIRST(math_type_pair) + "_x_grad"))        \
      .SetCreateFn<MathBinaryElementwiseXGradGpuKernel<                                         \
          OF_PP_CAT(OF_PP_PAIR_SECOND(math_type_pair), Functor),                                \
          OF_PP_PAIR_FIRST(data_type_pair)>>()                                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                          \
                       && (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(data_type_pair))); \
  REGISTER_USER_KERNEL((std::string("") + OF_PP_PAIR_FIRST(math_type_pair) + "_y_grad"))        \
      .SetCreateFn<MathBinaryElementwiseYGradGpuKernel<                                         \
          OF_PP_CAT(OF_PP_PAIR_SECOND(math_type_pair), Functor),                                \
          OF_PP_PAIR_FIRST(data_type_pair)>>()                                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                          \
                       && (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(data_type_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_ELEMENTWISE_CUDA_KERNEL_AND_GRAD,
                                 MATH_BINARY_ELEMENTWISE_FUNC_SEQ, FLOATING_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_ELEMENTWISE_CUDA_KERNEL_AND_GRAD,
                                 OF_PP_MAKE_TUPLE_SEQ("floordiv", FloorDiv),
                                 INT_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ)

template<template<typename> class BinaryFunctor>
class MathBinaryElementwiseGpuHalfKernel final : public user_op::OpKernel {
 public:
  MathBinaryElementwiseGpuHalfKernel() = default;
  ~MathBinaryElementwiseGpuHalfKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
    const half* x = reinterpret_cast<const half*>(tensor_x->dptr<float16>());
    const half* y = reinterpret_cast<const half*>(tensor_y->dptr<float16>());
    half* z = reinterpret_cast<half*>(tensor_z->mut_dptr<float16>());
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    if (n == 0) { return; }
    MathBinaryElementwiseForwardGpu<BinaryFunctor, half>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(n, x, y, z);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename> class BinaryFunctor>
class MathBinaryElementwiseXGradGpuHalfKernel final : public user_op::OpKernel {
 public:
  MathBinaryElementwiseXGradGpuHalfKernel() = default;
  ~MathBinaryElementwiseXGradGpuHalfKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const half* x = reinterpret_cast<const half*>(tensor_x->dptr<float16>());
    const half* y = reinterpret_cast<const half*>(tensor_y->dptr<float16>());
    const half* dz = reinterpret_cast<const half*>(tensor_dz->dptr<float16>());
    half* dx = reinterpret_cast<half*>(tensor_dx->mut_dptr<float16>());
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    if (n == 0) { return; }
    MathBinaryElementwiseBackwardXGradGpu<BinaryFunctor, half>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(n, x, y, dz, dx);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename> class BinaryFunctor>
class MathBinaryElementwiseYGradGpuHalfKernel final : public user_op::OpKernel {
 public:
  MathBinaryElementwiseYGradGpuHalfKernel() = default;
  ~MathBinaryElementwiseYGradGpuHalfKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);

    const half* x = reinterpret_cast<const half*>(tensor_x->dptr<float16>());
    const half* y = reinterpret_cast<const half*>(tensor_y->dptr<float16>());
    const half* dz = reinterpret_cast<const half*>(tensor_dz->dptr<float16>());
    half* dy = reinterpret_cast<half*>(tensor_dy->mut_dptr<float16>());
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    if (n == 0) { return; }
    MathBinaryElementwiseBackwardYGradGpu<BinaryFunctor, half>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(n, x, y, dz, dy);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MATH_BINARY_ELEMENTWISE_CUDA_HALF_KERNEL_AND_GRAD(math_type_str,              \
                                                                   math_func_prefix)           \
  REGISTER_USER_KERNEL(math_type_str)                                                          \
      .SetCreateFn<MathBinaryElementwiseGpuHalfKernel<OF_PP_CAT(math_func_prefix, Functor)>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                         \
                       && (user_op::HobDataType("x", 0) == DataType::kFloat16));               \
                                                                                               \
  REGISTER_USER_KERNEL((std::string("") + math_type_str + "_x_grad"))                          \
      .SetCreateFn<                                                                            \
          MathBinaryElementwiseXGradGpuHalfKernel<OF_PP_CAT(math_func_prefix, Functor)>>()     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                         \
                       && (user_op::HobDataType("x", 0) == DataType::kFloat16));               \
  REGISTER_USER_KERNEL((std::string("") + math_type_str + "_y_grad"))                          \
      .SetCreateFn<                                                                            \
          MathBinaryElementwiseYGradGpuHalfKernel<OF_PP_CAT(math_func_prefix, Functor)>>()     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                         \
                       && (user_op::HobDataType("x", 0) == DataType::kFloat16));

OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_ELEMENTWISE_CUDA_HALF_KERNEL_AND_GRAD,
                     MATH_BINARY_ELEMENTWISE_FUNC_SEQ)

}  // namespace oneflow
