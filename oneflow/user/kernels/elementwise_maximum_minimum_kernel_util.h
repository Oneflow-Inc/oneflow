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
#ifndef _ONEFLOW_USER_KERNEL_ELEMENTWISE_MAXIMUM_MINIMUM_KERNEL_UTIL_H
#define _ONEFLOW_USER_KERNEL_ELEMENTWISE_MAXIMUM_MINIMUM_KERNEL_UTIL_H
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace user_op {

template<typename T>
struct MaximumUtil {
  OF_DEVICE_FUNC T operator()(T x, T y) const { return x > y ? x : y; }

  OF_DEVICE_FUNC static void Backward(const T* dz, const T* x, const T* y, T* dx, T* dy) {
    const T dz_val = *dz;
    T dx_val = 0;
    T dy_val = 0;
    if (*x > *y) {
      dx_val = dz_val;
    } else {
      dy_val = dz_val;
    }
    if (dx) { *dx = dx_val; }
    if (dy) { *dy = dy_val; }
  }
};

template<typename T>
struct MinimumUtil {
  OF_DEVICE_FUNC T operator()(T x, T y) const { return x < y ? x : y; }

  OF_DEVICE_FUNC static void Backward(const T* dz, const T* x, const T* y, T* dx, T* dy) {
    const T dz_val = *dz;
    T dx_val = 0;
    T dy_val = 0;
    if (*x < *y) {
      dx_val = dz_val;
    } else {
      dy_val = dz_val;
    }
    if (dx) { *dx = dx_val; }
    if (dy) { *dy = dy_val; }
  }
};

template<DeviceType device_type, template<typename> class XmumUtil, typename T>
struct RunKernelUtil final {
  static void ForwardKernel(DeviceCtx* ctx, int64_t elem_cnt, T* z, const T* x, const T* y);
  static void BackwardKernel(DeviceCtx* ctx, int64_t elem_cnt, const T* dz, const T* x, const T* y,
                             T* dx, T* dy);
};

template<DeviceType device_type, template<typename> class XmumUtil, typename T>
class ElementwiseMaximumMinimumKernel final : public user_op::OpKernel {
 public:
  ElementwiseMaximumMinimumKernel() = default;
  ~ElementwiseMaximumMinimumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
    int64_t n = tensor_x->shape().elem_cnt();

    RunKernelUtil<device_type, XmumUtil, T>::ForwardKernel(
        ctx->device_ctx(), n, tensor_z->mut_dptr<T>(), tensor_x->dptr<T>(), tensor_y->dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, template<typename> class XmumUtil, typename T>
class ElementwiseMaximumMinimumBackwardKernel final : public user_op::OpKernel {
 public:
  ElementwiseMaximumMinimumBackwardKernel() = default;
  ~ElementwiseMaximumMinimumBackwardKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);

    const T* dptr_dz = tensor_dz->dptr<T>();
    const T* dptr_x = tensor_x->dptr<T>();
    const T* dptr_y = tensor_y->dptr<T>();

    T* dptr_dx = tensor_dx ? tensor_dx->mut_dptr<T>() : nullptr;
    T* dptr_dy = tensor_dy ? tensor_dy->mut_dptr<T>() : nullptr;

    RunKernelUtil<device_type, XmumUtil, T>::BackwardKernel(ctx->device_ctx(),
                                                            tensor_dz->shape().elem_cnt(), dptr_dz,
                                                            dptr_x, dptr_y, dptr_dx, dptr_dy);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace user_op
}  // namespace oneflow

#define REGISTER_FORWARD_KERNEL(dev_type, op_type_name, util, dtype)                 \
  REGISTER_USER_KERNEL(op_type_name)                                                 \
      .SetCreateFn<ElementwiseMaximumMinimumKernel<dev_type, util, dtype>>()         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == dev_type)                         \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#define REGISTER_BACKWARD_KERNEL(dev_type, op_type_name, util, dtype)                \
  REGISTER_USER_KERNEL(std::string("") + op_type_name + "_backward")                 \
      .SetCreateFn<ElementwiseMaximumMinimumBackwardKernel<dev_type, util, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == dev_type)                         \
                       & (user_op::HobDataType("dz", 0) == GetDataType<dtype>::value));

#endif  // _ONEFLOW_USER_KERNEL_ELEMENTWISE_MAXIMUM_MINIMUM_KERNEL_UTIL_H
