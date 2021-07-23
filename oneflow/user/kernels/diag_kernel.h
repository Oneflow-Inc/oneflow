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
#ifndef _ONEFLOW_USER_KERNELS_DIAG_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_DIAG_KERNEL_H_
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {
template<DeviceType device_type, typename T>
struct DiagFunctor final {
  void operator()(DeviceCtx* ctx, T* out_buf, const T* in_buf, int32_t size, int32_t stride,
                  int32_t in_dim);
};

template<DeviceType device_type, typename T>
struct DiagGradFunctor final {
  void operator()(DeviceCtx* ctx, T* dx_buf, const T* dy_buf, int32_t dx_cnt, int32_t dy_cnt,
                  int32_t stride, int32_t in_dim);
};
}  // namespace

template<DeviceType device_type, typename T>
class DiagKernel final : public user_op::OpKernel {
 public:
  DiagKernel() = default;
  ~DiagKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int32_t diagonal = ctx->Attr<int32_t>("diagonal");
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& out_shape = out->shape();
    const ShapeView& in_shape = in->shape();
    int32_t in_dim = in_shape.NumAxes();
    const T* in_buf = in->dptr<T>();
    T* out_buf = out->mut_dptr<T>();

    Memset<device_type>(ctx->device_ctx(), out->mut_dptr(), 0, out_shape.elem_cnt() * sizeof(T));

    if (in_dim == 1) {
      int32_t size = in_shape.elem_cnt();
      out_buf += (diagonal >= 0 ? diagonal : -diagonal * out_shape.At(1));
      DiagFunctor<device_type, T>()(ctx->device_ctx(), out_buf, in_buf, size, out_shape.At(1) + 1,
                                    in_dim);
    } else {
      int32_t size = 0;
      in_buf += (diagonal >= 0 ? diagonal : -diagonal * in_shape.At(1));
      if (diagonal >= 0) {
        size = std::min(in_shape.At(0), in_shape.At(1) - diagonal);
      } else {
        size = std::min(in_shape.At(0) + diagonal, in_shape.At(1));
      }
      DiagFunctor<device_type, T>()(ctx->device_ctx(), out_buf, in_buf, size, in_shape.At(1) + 1,
                                    in_dim);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class DiagBackwardKernel final : public user_op::OpKernel {
 public:
  DiagBackwardKernel() = default;
  ~DiagBackwardKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    int32_t diagonal = ctx->Attr<int32_t>("diagonal");
    const ShapeView& dx_shape = dx->shape();
    const ShapeView& dy_shape = dy->shape();
    int32_t in_dim = dx_shape.NumAxes();
    int32_t dy_cnt = dy_shape.Count(0);
    int32_t dx_cnt = dx_shape.Count(0);
    T* dx_buf = dx->mut_dptr<T>();
    const T* dy_buf = dy->dptr<T>();

    Memset<device_type>(ctx->device_ctx(), dx->mut_dptr<T>(), 0, dx_shape.elem_cnt() * sizeof(T));

    if (in_dim == 1) {
      dy_buf += (diagonal >= 0 ? diagonal : -diagonal * dy_shape.At(1));
      DiagGradFunctor<device_type, T>()(ctx->device_ctx(), dx_buf, dy_buf, dx_cnt, dy_cnt,
                                        dy_shape.At(1) + 1, in_dim);
    } else {
      dx_buf += (diagonal >= 0 ? diagonal : -diagonal * dx_shape.At(1));
      DiagGradFunctor<device_type, T>()(ctx->device_ctx(), dx_buf, dy_buf, dx_cnt, dy_cnt,
                                        dx_shape.At(1) + 1, in_dim);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DIAG_KERNELS(device, dtype)                                             \
  REGISTER_USER_KERNEL("diag").SetCreateFn<DiagKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device)                                                \
      & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));                   \
  REGISTER_USER_KERNEL("diag_grad")                                                      \
      .SetCreateFn<DiagBackwardKernel<device, dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_DIAG_KERNEL_H_
