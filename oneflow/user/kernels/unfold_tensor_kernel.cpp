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
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/user/kernels/unfold_tensor_kernel_utils.h"

namespace oneflow {

template<typename T>
class UnfoldTensorKernel final : public user_op::OpKernel {
 public:
  UnfoldTensorKernel() = default;
  ~UnfoldTensorKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("y", 0);

    const ShapeView& in_shape = in->shape_view();
    std::vector<int32_t> out_shape;
    out_shape.resize(out->shape_view().NumAxes());
    for (int i = 0; i < out->shape_view().NumAxes(); ++i) {
      out_shape[i] = out->shape_view().At(i);
    }

    const int32_t in_dims = in_shape.NumAxes();
    const int32_t out_dims = out_shape.size();
    const int32_t dimension = ctx->Attr<int32_t>("dimension");
    const int32_t step = ctx->Attr<int32_t>("step");

    std::vector<int32_t> in_stride(in_dims, 1);
    for (int32_t i = in_dims - 2; i >= 0; --i) {
      in_stride[i] = in_shape.At(i + 1) * in_stride.at(i + 1);
    }

    std::vector<int32_t> out_stride(in_dims + 1);
    out_stride[in_dims] = in_dims == 0 ? 1 : in_stride[dimension];
    for (int d = 0; d < in_dims; ++d) {
      if (d == dimension) {
        out_stride[d] = step * in_stride[d];
      } else {
        out_stride[d] = in_stride[d];
      }
    }

    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();
    const int32_t out_size = out->shape_view().elem_cnt();
    for (int32_t i = 0; i < out_size; ++i) {
      int offset = Offset(i, out_stride.data(), out_shape.data(), out_dims - 1);
      out_ptr[i] = in_ptr[offset];
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UNFOLD_TENSOR_KERNEL(dtype)                          \
  REGISTER_USER_KERNEL("unfold_tensor")                               \
      .SetCreateFn<UnfoldTensorKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_UNFOLD_TENSOR_KERNEL(float)
REGISTER_UNFOLD_TENSOR_KERNEL(double)
REGISTER_UNFOLD_TENSOR_KERNEL(int64_t)
REGISTER_UNFOLD_TENSOR_KERNEL(int32_t)

template<typename T>
class UnfoldTensorGradKernel final : public user_op::OpKernel {
 public:
  UnfoldTensorGradKernel() = default;
  ~UnfoldTensorGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dout = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* din = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const ShapeView& in_shape = in->shape_view();
    const int32_t in_dims = in_shape.NumAxes();
    std::vector<int32_t> din_stride(in_dims, 1);
    for (int32_t i = in_dims - 2; i >= 0; --i) {
      din_stride[i] = in_shape.At(i + 1) * din_stride.at(i + 1);
    }

    std::vector<int32_t> dout_shape;
    dout_shape.resize(dout->shape_view().NumAxes());
    for (int i = 0; i < dout->shape_view().NumAxes(); ++i) {
      dout_shape[i] = dout->shape_view().At(i);
    }

    const int32_t dout_dims = dout_shape.size();
    const int32_t dimension = ctx->Attr<int32_t>("dimension");
    const int32_t step = ctx->Attr<int32_t>("step");

    std::vector<int32_t> dout_stride(in_dims + 1);
    dout_stride[in_dims] = in_dims == 0 ? 1 : din_stride[dimension];
    for (int d = 0; d < in_dims; ++d) {
      if (d == dimension) {
        dout_stride[d] = step * din_stride[d];
      } else {
        dout_stride[d] = din_stride[d];
      }
    }

    const T* dout_ptr = dout->dptr<T>();
    T* din_ptr = din->mut_dptr<T>();

    std::fill(din_ptr, din_ptr + din->shape_view().elem_cnt(), static_cast<T>(0));
    const int32_t dout_size = dout->shape_view().elem_cnt();
    for (int32_t i = 0; i < dout_size; ++i) {
      int offset = Offset(i, dout_stride.data(), dout_shape.data(), dout_dims - 1);
      din_ptr[offset] += dout_ptr[i];
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UNFOLD_TENSOR_GRAD_KERNEL(dtype)                     \
  REGISTER_USER_KERNEL("unfold_tensor_grad")                          \
      .SetCreateFn<UnfoldTensorGradKernel<dtype>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_UNFOLD_TENSOR_GRAD_KERNEL(float)
REGISTER_UNFOLD_TENSOR_GRAD_KERNEL(double)
REGISTER_UNFOLD_TENSOR_GRAD_KERNEL(int64_t)
REGISTER_UNFOLD_TENSOR_GRAD_KERNEL(int32_t)

}  // namespace oneflow
