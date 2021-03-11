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
  void operator()(DeviceCtx* ctx, T* out_buf, const T* in_buf, int32_t size, int32_t strideSum,
                  int32_t in_dim);
};

template<DeviceType device_type, typename T>
struct DiagGradFunctor final {
  void operator()(DeviceCtx* ctx, T* dx_buf, const T* dy_buf, int32_t dx_num_cnt,
                  int32_t dy_num_cnt, int32_t strideSum, int32_t in_dim);
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

    Memset<device_type>(ctx->device_ctx(), out->mut_dptr(), 0, out_shape.elem_cnt() * sizeof(T));

    const T* in_buf = in->dptr<T>();
    T* out_buf = out->mut_dptr<T>();

    int32_t stride_0 = 0;
    int32_t stride_1 = 0;
    int32_t sz = 0;
    if (in_dim == 1) {
      stride_0 = out_shape.At(1);
      stride_1 = 1;
      sz = in_shape.elem_cnt();
      out_buf += (diagonal >= 0 ? diagonal * stride_1 : -diagonal * stride_0);
    } else {
      stride_0 = in_shape.At(1);
      stride_1 = 1;
      in_buf += (diagonal >= 0 ? diagonal * stride_1 : -diagonal * stride_0);
      if (diagonal >= 0) {
        sz = std::min(in_shape.At(0), in_shape.At(1) - diagonal);
      } else {
        sz = std::min(in_shape.At(0) + diagonal, in_shape.At(1));
      }
    }
    DiagFunctor<device_type, T>()(ctx->device_ctx(), out_buf, in_buf, sz, stride_0 + stride_1,
                                  in_dim);
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
    int32_t dy_num_cnt = dy_shape.At(0);
    int32_t dx_num_cnt = dx_shape.Count(0);
    T* dx_buf = dx->mut_dptr<T>();
    const T* dy_buf = dy->dptr<T>();

    Memset<device_type>(ctx->device_ctx(), dx->mut_dptr<T>(), 0, dx_shape.elem_cnt() * sizeof(T));

    int32_t stride_1 = 0;
    int32_t stride_0 = 0;
    if (in_dim == 1) {
      stride_1 = 1;
      stride_0 = dy_shape.At(1);

      dy_buf += (diagonal >= 0 ? diagonal * stride_1 : -diagonal * stride_0);
      for (int32_t i = 0; i < dx_num_cnt; i++) { dx_buf[i] = dy_buf[i * (stride_0 + stride_1)]; }
    } else {
      stride_0 = dx_shape.At(1);
      stride_1 = 1;
      dx_buf += (diagonal >= 0 ? diagonal * stride_1 : -diagonal * stride_0);
      for (int32_t i = 0; i < dy_num_cnt; i++) { dx_buf[i * (stride_0 + stride_1)] = dy_buf[i]; }
    }
    DiagGradFunctor<device_type, T>()(ctx->device_ctx(), dx_buf, dy_buf, dx_num_cnt, dy_num_cnt,
                                      stride_0 + stride_1, in_dim);
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
