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
#include "oneflow/user/kernels/greater_inplace_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class GreaterInplaceKernel final : public user_op::OpKernel {
 public:
  GreaterInplaceKernel() = default;
  ~GreaterInplaceKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const int64_t elem_cnt = x->shape_view().elem_cnt();
    if (elem_cnt == 0) { return; }
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const T* x_ptr = x->dptr<T>();
    const T* y_ptr = y->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();
    T* broadcast_y_ptr = tmp_buffer->mut_dptr<T>();

    if (x->shape_view() == y->shape_view()) {
      GreaterInplaceKernelUtil<device_type, T>::Forward(ctx->stream(), elem_cnt, x_ptr, y_ptr,
                                                        out_ptr);
      return;
    }
    GreaterInplaceKernelUtil<device_type, T>::YBroadcastToX(
        ctx->stream(), elem_cnt, x_ptr, y_ptr, broadcast_y_ptr, x->shape_view(), y->shape_view());
    GreaterInplaceKernelUtil<device_type, T>::Forward(ctx->stream(), elem_cnt, x_ptr,
                                                      broadcast_y_ptr, out_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GREATER_INPLACE_KERNEL(device_type, dtype)                              \
  REGISTER_USER_KERNEL("broadcast_inplace_greater")                                      \
      .SetCreateFn<GreaterInplaceKernel<device_type, dtype>>()                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == device_type)                         \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                \
        const Shape& x_shape = ctx->InputShape("x", 0);                                  \
        return GetCudaAlignedSize(x_shape.elem_cnt() * sizeof(dtype));                   \
      });

REGISTER_GREATER_INPLACE_KERNEL(DeviceType::kCPU, float)
REGISTER_GREATER_INPLACE_KERNEL(DeviceType::kCPU, double)
REGISTER_GREATER_INPLACE_KERNEL(DeviceType::kCPU, int8_t)
REGISTER_GREATER_INPLACE_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_GREATER_INPLACE_KERNEL(DeviceType::kCPU, int64_t)

#ifdef WITH_CUDA
REGISTER_GREATER_INPLACE_KERNEL(DeviceType::kCUDA, half)
REGISTER_GREATER_INPLACE_KERNEL(DeviceType::kCUDA, float)
REGISTER_GREATER_INPLACE_KERNEL(DeviceType::kCUDA, double)
REGISTER_GREATER_INPLACE_KERNEL(DeviceType::kCUDA, int8_t)
REGISTER_GREATER_INPLACE_KERNEL(DeviceType::kCUDA, int32_t)
REGISTER_GREATER_INPLACE_KERNEL(DeviceType::kCUDA, int64_t)
#endif  // WITH_CUDA

template<DeviceType device_type, typename T, typename ValueT>
class ScalarGreaterInplaceKernel final : public user_op::OpKernel {
 public:
  ScalarGreaterInplaceKernel() = default;
  ~ScalarGreaterInplaceKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const int64_t elem_cnt = in->shape_view().elem_cnt();
    if (elem_cnt == 0) { return; }
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Scalar scalar_operand;
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = ctx->Attr<int64_t>("int_operand");
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = ctx->Attr<double>("float_operand");
    } else {
      UNIMPLEMENTED();
    }

    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();

    ScalarGreaterInplaceKernelUtil<device_type, T, ValueT>::Forward(ctx->stream(), elem_cnt, in_ptr,
                                                                    scalar_operand, out_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SCALAR_GREATER_INPLACE_KERNEL(device_type, dtype, value_type)   \
  REGISTER_USER_KERNEL("scalar_logical_inplace_greater")                         \
      .SetCreateFn<ScalarGreaterInplaceKernel<device_type, dtype, value_type>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device_type)                 \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_SCALAR_GREATER_INPLACE_KERNEL(DeviceType::kCPU, float, double)
REGISTER_SCALAR_GREATER_INPLACE_KERNEL(DeviceType::kCPU, double, double)
REGISTER_SCALAR_GREATER_INPLACE_KERNEL(DeviceType::kCPU, int8_t, int64_t)
REGISTER_SCALAR_GREATER_INPLACE_KERNEL(DeviceType::kCPU, int32_t, int64_t)
REGISTER_SCALAR_GREATER_INPLACE_KERNEL(DeviceType::kCPU, int64_t, int64_t)

#ifdef WITH_CUDA
REGISTER_SCALAR_GREATER_INPLACE_KERNEL(DeviceType::kCUDA, half, double)
REGISTER_SCALAR_GREATER_INPLACE_KERNEL(DeviceType::kCUDA, float, double)
REGISTER_SCALAR_GREATER_INPLACE_KERNEL(DeviceType::kCUDA, double, double)
REGISTER_SCALAR_GREATER_INPLACE_KERNEL(DeviceType::kCUDA, int8_t, int64_t)
REGISTER_SCALAR_GREATER_INPLACE_KERNEL(DeviceType::kCUDA, int32_t, int64_t)
REGISTER_SCALAR_GREATER_INPLACE_KERNEL(DeviceType::kCUDA, int64_t, int64_t)
#endif  // WITH_CUDA

}  // namespace oneflow
