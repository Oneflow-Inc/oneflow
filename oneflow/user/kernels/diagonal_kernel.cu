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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace {

template<typename T>
__global__ void forward_diagonal_kernel(T* out_buf, const T* in_buf, int32_t count_dim, int32_t dim1,
                                        int32_t dim2, int32_t last_dim) {
  int32_t offset1 = dim1 * dim2;
  int32_t offset2 = dim2 + 1;
  CUDA_1D_KERNEL_LOOP(index, count_dim * last_dim) {
    int32_t i = index / last_dim;
    int32_t j = index - i * last_dim;
    out_buf[index] = in_buf[i * offset1 + j * offset2];
  }
}

template<typename T>
__global__ void backward_diagonal_kernel(T* dx_buf, const T* dy_buf, int32_t count_dim, int32_t dim1,
                                        int32_t dim2, int32_t last_dim) {
  int32_t offset1 = dim1 * dim2;
  int32_t offset2 = dim2 + 1;
  CUDA_1D_KERNEL_LOOP(index, count_dim * last_dim) {
    int32_t i = index / last_dim;
    int32_t j = index - i * last_dim;
    dx_buf[i * offset1 + j * offset2] = dy_buf[index];
  }
}

template<typename T>
struct DiagonalFunctor final {
  void operator()(ep::Stream* stream, T* out_buf, const T* in_buf, int32_t count_dim, int32_t dim1,
                  int32_t dim2, int32_t last_dim) {
    if (count_dim * last_dim > 0) {
      forward_diagonal_kernel<T>
          <<<BlocksNum4ThreadsNum(count_dim * last_dim), kCudaThreadsNumPerBlock, 0,
             stream->As<ep::CudaStream>()->cuda_stream()>>>(out_buf, in_buf, count_dim, dim1, dim2,last_dim);
    }
  }
};

template<typename T>
struct DiagonalGradFunctor final {
  void operator()(ep::Stream* stream, T* dx_buf, const T* dy_buf, int32_t count_dim, int32_t dim1,
                  int32_t dim2, int32_t last_dim) {
    if (count_dim * last_dim > 0) {
      backward_diagonal_kernel<T>
          <<<BlocksNum4ThreadsNum(count_dim * last_dim ), kCudaThreadsNumPerBlock, 0,
             stream->As<ep::CudaStream>()->cuda_stream()>>>(dx_buf, dy_buf, count_dim, dim1, dim2,last_dim);
    }
  }
};

}  // namespace

template<typename T>
class GpuDiagonalKernel final : public user_op::OpKernel {
 public:
  GpuDiagonalKernel() = default;
  ~GpuDiagonalKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int32_t offset = ctx->Attr<int32_t>("offset");
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& out_shape = out->shape();
    const ShapeView& in_shape = in->shape();
    const T* in_buf = in->dptr<T>();
    T* out_buf = out->mut_dptr<T>();
    int32_t in_dim = in_shape.NumAxes();
    int32_t out_dim = out_shape.NumAxes();
    
    int32_t count_dim = in_dim<=2 ? 1 : in_shape.Count(0 , in_dim - 2);
    int32_t dim1 = in_shape.At(in_dim-2);
    int32_t dim2 = in_shape.At(in_dim-1);
    int32_t last_dim = out_shape.At(out_dim - 1);

    int32_t offset_in_bufer = (offset >= 0 ? offset : -offset * dim2);
    in_buf += offset_in_bufer;
    DiagonalFunctor<T>()(ctx->stream(), out_buf, in_buf, count_dim, dim1, dim2, last_dim);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class GpuDiagonalBackwardKernel final : public user_op::OpKernel {
 public:
  GpuDiagonalBackwardKernel() = default;
  ~GpuDiagonalBackwardKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
   const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    int32_t offset = ctx->Attr<int32_t>("offset");
    const ShapeView& dx_shape = dx->shape();
    const ShapeView& dy_shape = dy->shape();
    T* dx_buf = dx->mut_dptr<T>();
    const T* dy_buf = dy->dptr<T>();
    int32_t dx_dim = dx_shape.NumAxes();
    int32_t dy_dim = dy_shape.NumAxes();
    
    int32_t count_dim = dx_dim <= 2 ? 1 : dx_shape.Count(0 , dx_dim - 2);
    int32_t dim1 = dx_shape.At(dx_dim-2);
    int32_t dim2 = dx_shape.At(dx_dim-1);
    int32_t last_dim = dy_shape.At(dy_dim - 1);
    
    int32_t offset_dx_bufer = (offset >= 0 ? offset : -offset * dim2);

    Memset<DeviceType::kCUDA>(ctx->stream(), dx->mut_dptr<T>(), 0, dx_shape.elem_cnt() * sizeof(T));
    dx_buf += offset_dx_bufer;
    DiagonalGradFunctor<T>()(ctx->stream(), dx_buf, dy_buf, count_dim, dim1, dim2, last_dim);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DIAGONAL_KERNELS(dtype)                                                 \
  REGISTER_USER_KERNEL("diagonal")                                                       \
      .SetCreateFn<GpuDiagonalKernel<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                   \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("diagonal_grad")                                                  \
      .SetCreateFn<GpuDiagonalBackwardKernel<dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                   \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_DIAGONAL_KERNELS(bool);
REGISTER_DIAGONAL_KERNELS(half);
REGISTER_DIAGONAL_KERNELS(float);
REGISTER_DIAGONAL_KERNELS(double);
REGISTER_DIAGONAL_KERNELS(int8_t);
REGISTER_DIAGONAL_KERNELS(int32_t);
REGISTER_DIAGONAL_KERNELS(int64_t);

#undef REGISTER_DIAGONAL_KERNELS

}  // namespace oneflow
