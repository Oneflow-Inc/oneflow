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
#include "oneflow/user/kernels/expand_kernel_utils.h"

#include <algorithm>

namespace oneflow {

template<typename T>
class CpuExpandKernel final : public user_op::OpKernel {
 public:
  CpuExpandKernel() = default;
  ~CpuExpandKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const std::vector<int32_t>& logical_expand_shape =
        ctx->Attr<std::vector<int32_t>>("logical_expand_shape");
    if (std::any_of(logical_expand_shape.begin(), logical_expand_shape.end(),
                    [](int32_t dim_size) { return dim_size == 0; })) {
      return;
    }
    std::vector<int32_t> in_shape;
    in_shape.resize(in->shape().NumAxes());
    for (int i = 0; i < in->shape().NumAxes(); ++i) { in_shape[i] = in->shape().At(i); }

    std::vector<int32_t> out_shape;
    std::vector<int32_t> expand_stride;
    CHECK_JUST(getOutShapeAndStrideForFp(in_shape, logical_expand_shape, out_shape, expand_stride));

    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();
    const int32_t out_dims = out->shape().NumAxes();
    const int32_t out_size = out->shape().elem_cnt();
    int32_t out_stride[out_dims];
    InitStride(out_stride, out_shape.data(), out_dims);
    for (int32_t i = 0; i < out_size; ++i) {
      int offset = OffsetToNdIndexToOffset(i, out_stride, expand_stride.data(), out_dims);
      out_ptr[i] = in_ptr[offset];
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_EXPAND_KERNEL(dtype)                                                   \
  REGISTER_USER_KERNEL("expand").SetCreateFn<CpuExpandKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                    \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_EXPAND_KERNEL(float);
REGISTER_EXPAND_KERNEL(double);
REGISTER_EXPAND_KERNEL(bool);
REGISTER_EXPAND_KERNEL(uint8_t);
REGISTER_EXPAND_KERNEL(int8_t);
REGISTER_EXPAND_KERNEL(int32_t);
REGISTER_EXPAND_KERNEL(int64_t);

template<typename T>
class CpuExpandGradKernel final : public user_op::OpKernel {
 public:
  CpuExpandGradKernel() = default;
  ~CpuExpandGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const std::vector<int32_t>& logical_out_shape =
        ctx->Attr<std::vector<int32_t>>("logical_out_shape");
    const std::vector<int32_t>& logical_expand_shape =
        ctx->Attr<std::vector<int32_t>>("logical_expand_shape");

    std::vector<int32_t> in_shape;
    in_shape.resize(in->shape().NumAxes());
    for (int i = 0; i < in->shape().NumAxes(); ++i) { in_shape[i] = in->shape().At(i); }
    std::vector<int32_t> out_shape;
    std::vector<int32_t> expand_stride;
    CHECK_JUST(getOutShapeAndStrideForBp(logical_out_shape, logical_expand_shape, in_shape,
                                         out_shape, expand_stride));

    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();

    const int32_t in_dims = in->shape().NumAxes();
    const int32_t in_size = in->shape().elem_cnt();
    int32_t in_stride[in_dims];
    InitStride(in_stride, in_shape.data(), in_dims);

    std::fill(out_ptr, out_ptr + out->shape().elem_cnt(), static_cast<T>(0));
    for (int i = 0; i < in_size; ++i) {
      int offset = OffsetToNdIndexToOffset(i, in_stride, expand_stride.data(), in_dims);
      out_ptr[offset] += in_ptr[i];
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_EXPAND_GRAD_KERNEL(dtype)                            \
  REGISTER_USER_KERNEL("expand_grad")                                 \
      .SetCreateFn<CpuExpandGradKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_EXPAND_GRAD_KERNEL(float);
REGISTER_EXPAND_GRAD_KERNEL(double);
REGISTER_EXPAND_GRAD_KERNEL(int32_t);
REGISTER_EXPAND_GRAD_KERNEL(int64_t);

}  // namespace oneflow
