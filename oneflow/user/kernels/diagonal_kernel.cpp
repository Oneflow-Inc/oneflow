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
struct DiagonalFunctor final {
  void operator()(ep::Stream* stream, T* out_buf, const T* in_buf, int32_t size, int32_t dim1,
                  int32_t dim2) {
    int32_t offset_index = (dim1 + 1) * dim2;
    FOR_RANGE(int32_t, index, 0, size * dim2) {
      int32_t i = index / dim2;
      int32_t j = index - i * dim2;
      out_buf[j * size + i] = in_buf[i * offset_index + j];
    }
  }
};

template<typename T>
struct DiagonalGradFunctor final {
  void operator()(ep::Stream* stream, T* dx_buf, const T* dy_buf, int32_t size, int32_t dim1,
                  int32_t dim2) {
    int32_t offset_index = (dim1 + 1) * dim2;
    FOR_RANGE(int32_t, index, 0, size * dim2) {
      int32_t i = index / dim2;
      int32_t j = index - i * dim2;
      dx_buf[i * offset_index + j] = dy_buf[j * size + i];
    }
  }
};

}  // namespace

template<typename T>
class CpuDiagonalKernel final : public user_op::OpKernel {
 public:
  CpuDiagonalKernel() = default;
  ~CpuDiagonalKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int32_t offset = ctx->Attr<int32_t>("offset");
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& out_shape = out->shape_view();
    const ShapeView& in_shape = in->shape_view();
    const T* in_buf = in->dptr<T>();
    T* out_buf = out->mut_dptr<T>();

    int32_t size = out_shape.At(out_shape.NumAxes() - 1);
    int32_t dim1 = in_shape.At(1);
    int32_t dim2 = 0;
    if (in_shape.NumAxes() <= 2) {
      dim2 = 1;
    } else {
      dim2 = in_shape.Count(2, in_shape.NumAxes());
    }

    int32_t offset_in_bufer = (offset >= 0 ? offset * dim2 : -offset * dim1 * dim2);
    in_buf += offset_in_bufer;
    DiagonalFunctor<T>()(ctx->stream(), out_buf, in_buf, size, dim1, dim2);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class CpuDiagonalBackwardKernel final : public user_op::OpKernel {
 public:
  CpuDiagonalBackwardKernel() = default;
  ~CpuDiagonalBackwardKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    int32_t offset = ctx->Attr<int32_t>("offset");
    const ShapeView& dx_shape = dx->shape_view();
    const ShapeView& dy_shape = dy->shape_view();
    T* dx_buf = dx->mut_dptr<T>();
    const T* dy_buf = dy->dptr<T>();

    Memset<DeviceType::kCPU>(ctx->stream(), dx->mut_dptr<T>(), 0, dx_shape.elem_cnt() * sizeof(T));

    int32_t dim1 = dx_shape.At(1);
    int32_t dim2 = 0;
    if (dx_shape.NumAxes() <= 2) {
      dim2 = 1;
    } else {
      dim2 = dx_shape.Count(2, dx_shape.NumAxes());
    }
    int32_t size = dy_shape.At(dy_shape.NumAxes() - 1);
    int32_t offset_in_bufer = (offset >= 0 ? offset * dim2 : -offset * dim1 * dim2);
    dx_buf += offset_in_bufer;

    DiagonalGradFunctor<T>()(ctx->stream(), dx_buf, dy_buf, size, dim1, dim2);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DIAGONAL_KERNELS(dtype)                                                 \
  REGISTER_USER_KERNEL("diagonal")                                                       \
      .SetCreateFn<CpuDiagonalKernel<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                    \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("diagonal_grad")                                                  \
      .SetCreateFn<CpuDiagonalBackwardKernel<dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                    \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_DIAGONAL_KERNELS(bool);
REGISTER_DIAGONAL_KERNELS(float);
REGISTER_DIAGONAL_KERNELS(double);
REGISTER_DIAGONAL_KERNELS(int8_t);
REGISTER_DIAGONAL_KERNELS(int32_t);
REGISTER_DIAGONAL_KERNELS(int64_t);

#undef REGISTER_DIAGONAL_KERNELS

}  // namespace oneflow
