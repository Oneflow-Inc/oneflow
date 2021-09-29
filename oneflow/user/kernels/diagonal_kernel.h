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
#ifndef _ONEFLOW_USER_KERNELS_DIAGONAL_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_DIAGONAL_KERNEL_H_
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {
template<DeviceType device_type, typename T>
struct DiagonalFunctor final {
  void operator()(DeviceCtx* ctx, T* out_buf, const T* in_buf, int32_t size, int32_t dim1,
                  int32_t dim2);
};

template<DeviceType device_type, typename T>
struct DiagonalGradFunctor final {
  void operator()(DeviceCtx* ctx, T* dx_buf, const T* dy_buf, int32_t size, int32_t dim1,
                  int32_t dim2);
};
}  // namespace

template<DeviceType device_type, typename T>
class DiagonalKernel final : public user_op::OpKernel {
 public:
  DiagonalKernel() = default;
  ~DiagonalKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int32_t offset = ctx->Attr<int32_t>("offset");
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& out_shape = out->shape();
    const ShapeView& in_shape = in->shape();
    const T* in_buf = in->dptr<T>();
    T* out_buf = out->mut_dptr<T>();

    Memset<device_type>(ctx->device_ctx(), out->mut_dptr(), 0, out_shape.elem_cnt() * sizeof(T));
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
    DiagonalFunctor<device_type, T>()(ctx->device_ctx(), out_buf, in_buf, size, dim1, dim2);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class DiagonalBackwardKernel final : public user_op::OpKernel {
 public:
  DiagonalBackwardKernel() = default;
  ~DiagonalBackwardKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    int32_t offset = ctx->Attr<int32_t>("offset");
    const ShapeView& dx_shape = dx->shape();
    const ShapeView& dy_shape = dy->shape();
    T* dx_buf = dx->mut_dptr<T>();
    const T* dy_buf = dy->dptr<T>();

    Memset<device_type>(ctx->device_ctx(), dx->mut_dptr<T>(), 0, dx_shape.elem_cnt() * sizeof(T));

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

    DiagonalGradFunctor<device_type, T>()(ctx->device_ctx(), dx_buf, dy_buf, size, dim1, dim2);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DIAGONAL_KERNELS(device, dtype)                                        \
  REGISTER_USER_KERNEL("diagonal")                                                      \
      .SetCreateFn<DiagonalKernel<device, dtype>>()                                     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("diagonal_grad")                                                 \
      .SetCreateFn<DiagonalBackwardKernel<device, dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_DIAGONAL_KERNEL_H_