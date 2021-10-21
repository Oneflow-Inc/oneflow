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
#ifndef _ONEFLOW_USER_KERNELS_ELEMENTWISE_MAXIMUM_MINIMUM_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_ELEMENTWISE_MAXIMUM_MINIMUM_KERNEL_H_
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
struct MaximumFunctor {
  OF_DEVICE_FUNC T operator()(const T x, const T y) const { return x > y ? x : y; }
};

template<typename T>
struct MaximumGradFunctor {
  OF_DEVICE_FUNC void operator()(const T dz, const T x, const T y, T* dx, T* dy) {
    T dx_val = 0;
    T dy_val = 0;
    if (x > y) {
      dx_val = dz;
    } else {
      dy_val = dz;
    }
    if (dx) { *dx = dx_val; }
    if (dy) { *dy = dy_val; }
  }
};

template<typename T>
struct MinimumFunctor {
  OF_DEVICE_FUNC T operator()(const T x, const T y) const { return x < y ? x : y; }
};

template<typename T>
struct MinimumGradFunctor {
  OF_DEVICE_FUNC void operator()(const T dz, const T x, const T y, T* dx, T* dy) {
    T dx_val = 0;
    T dy_val = 0;
    if (x < y) {
      dx_val = dz;
    } else {
      dy_val = dz;
    }
    if (dx) { *dx = dx_val; }
    if (dy) { *dy = dy_val; }
  }
};

namespace {
template<DeviceType device_type, template<typename> class Opt, typename T>
struct ElemwiseXimumGradFunctor final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, const T* dz, const T* x, const T* y, T* dx,
                  T* dy);
};

template<DeviceType device_type, template<typename> class Opt, typename T>
struct ElemwiseXimumFunctor final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, T* z, const T* x, const T* y);
};
}  // namespace

template<DeviceType device_type, template<typename> class Opt, typename T>
class ElemwiseXimumKernel final : public user_op::OpKernel {
 public:
  ElemwiseXimumKernel() = default;
  ~ElemwiseXimumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
    int64_t n = tensor_x->shape().elem_cnt();

    ElemwiseXimumFunctor<device_type, Opt, T>()(ctx->device_ctx(), n, tensor_z->mut_dptr<T>(),
                                                tensor_x->dptr<T>(), tensor_y->dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, template<typename> class Opt, typename T>
class ElemwiseXimumBackwardKernel final : public user_op::OpKernel {
 public:
  ElemwiseXimumBackwardKernel() = default;
  ~ElemwiseXimumBackwardKernel() = default;

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

    ElemwiseXimumGradFunctor<device_type, Opt, T>()(ctx->device_ctx(),
                                                    tensor_dz->shape().elem_cnt(), dptr_dz, dptr_x,
                                                    dptr_y, dptr_dx, dptr_dy);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MAXIMUM_KERNELS(device, dtype)                                        \
  REGISTER_USER_KERNEL("elementwise_maximum")                                          \
      .SetCreateFn<ElemwiseXimumKernel<device, MaximumFunctor, dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)   \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("elementwise_maximum_backward")                                 \
      .SetCreateFn<ElemwiseXimumBackwardKernel<device, MaximumGradFunctor, dtype>>()   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)   \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#define REGISTER_MINIMUM_KERNELS(device, dtype)                                        \
  REGISTER_USER_KERNEL("elementwise_minimum")                                          \
      .SetCreateFn<ElemwiseXimumKernel<device, MinimumFunctor, dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)   \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("elementwise_minimum_backward")                                 \
      .SetCreateFn<ElemwiseXimumBackwardKernel<device, MinimumGradFunctor, dtype>>()   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)   \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_ELEMENTWISE_MAXIMUM_MINIMUM_KERNEL_H_
