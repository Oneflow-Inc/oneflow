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
#include "oneflow/user/kernels/lerp_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class LerpKernel final : public user_op::OpKernel {
 public:
  LerpKernel() = default;
  ~LerpKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* start = ctx->Tensor4ArgNameAndIndex("start", 0);
    const user_op::Tensor* end = ctx->Tensor4ArgNameAndIndex("end", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const ShapeView& start_shape = start->shape_view();
    const ShapeView& end_shape = end->shape_view();
    const ShapeView& weight_shape = weight->shape_view();
    CHECK_EQ(start_shape, end_shape);
    CHECK_EQ(start_shape, weight_shape);

    const T* start_ptr = start->dptr<T>();
    const T* end_ptr = end->dptr<T>();
    const T* weight_ptr = weight->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();

    LerpKernelUtil<device_type, T>::Forward(ctx->stream(), start_shape.elem_cnt(), start_ptr,
                                            weight_ptr, end_ptr, out_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_LERP_KERNEL(device_type, dtype)                 \
  REGISTER_USER_KERNEL("lerp")                                   \
      .SetCreateFn<LerpKernel<device_type, dtype>>()             \
      .SetIsMatchedHob((user_op::HobDeviceType() == device_type) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_LERP_KERNEL(DeviceType::kCPU, float)
REGISTER_LERP_KERNEL(DeviceType::kCPU, double)
REGISTER_LERP_KERNEL(DeviceType::kCPU, uint8_t)
REGISTER_LERP_KERNEL(DeviceType::kCPU, int8_t)
REGISTER_LERP_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_LERP_KERNEL(DeviceType::kCPU, int64_t)

REGISTER_LERP_KERNEL(DeviceType::kCUDA, half)
REGISTER_LERP_KERNEL(DeviceType::kCUDA, float)
REGISTER_LERP_KERNEL(DeviceType::kCUDA, double)
REGISTER_LERP_KERNEL(DeviceType::kCUDA, uint8_t)
REGISTER_LERP_KERNEL(DeviceType::kCUDA, int8_t)
REGISTER_LERP_KERNEL(DeviceType::kCUDA, int32_t)
REGISTER_LERP_KERNEL(DeviceType::kCUDA, int64_t)

template<DeviceType device_type, typename T>
class LerpGradKernel final : public user_op::OpKernel {
 public:
  LerpGradKernel() = default;
  ~LerpGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* start = ctx->Tensor4ArgNameAndIndex("start", 0);
    const user_op::Tensor* end = ctx->Tensor4ArgNameAndIndex("end", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* out_diff = ctx->Tensor4ArgNameAndIndex("out_diff", 0);
    user_op::Tensor* start_diff = ctx->Tensor4ArgNameAndIndex("start_diff", 0);
    user_op::Tensor* end_diff = ctx->Tensor4ArgNameAndIndex("end_diff", 0);
    user_op::Tensor* weight_diff = ctx->Tensor4ArgNameAndIndex("weight_diff", 0);

    const ShapeView& start_shape = start->shape_view();
    const ShapeView& end_shape = end->shape_view();
    const ShapeView& weight_shape = weight->shape_view();
    CHECK_EQ(start_shape, end_shape);
    CHECK_EQ(start_shape, weight_shape);

    const T* start_ptr = start->dptr<T>();
    const T* end_ptr = end->dptr<T>();
    const T* weight_ptr = weight->dptr<T>();
    const T* out_diff_ptr = out_diff->dptr<T>();
    T* start_diff_ptr = start_diff->mut_dptr<T>();
    T* end_diff_ptr = end_diff->mut_dptr<T>();
    T* weight_diff_ptr = weight_diff->mut_dptr<T>();

    LerpKernelUtil<device_type, T>::Backward(ctx->stream(), start_shape.elem_cnt(), start_ptr,
                                             weight_ptr, end_ptr, out_diff_ptr, start_diff_ptr,
                                             weight_diff_ptr, end_diff_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_LERP_GRAD_KERNEL(device_type, dtype)            \
  REGISTER_USER_KERNEL("lerp_grad")                              \
      .SetCreateFn<LerpGradKernel<device_type, dtype>>()         \
      .SetIsMatchedHob((user_op::HobDeviceType() == device_type) \
                       && (user_op::HobDataType("start_diff", 0) == GetDataType<dtype>::value));

REGISTER_LERP_GRAD_KERNEL(DeviceType::kCPU, float)
REGISTER_LERP_GRAD_KERNEL(DeviceType::kCPU, double)

REGISTER_LERP_GRAD_KERNEL(DeviceType::kCUDA, half)
REGISTER_LERP_GRAD_KERNEL(DeviceType::kCUDA, float)
REGISTER_LERP_GRAD_KERNEL(DeviceType::kCUDA, double)

}  // namespace

}  // namespace oneflow