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

template<DeviceType device_type, template<typename> class GradFunctor, typename T>
struct RunKernelUtil final {
  static void ForwardKernel(DeviceCtx* ctx, int64_t elem_cnt, const T* dz, const T* x, const T* y, T* dx,
                  T* dy);
  static void BackwardKernel(DeviceCtx* ctx, int64_t elem_cnt, const T* dz, const T* x, const T* y, T* dx,
                  T* dy);
};

template<DeviceType device_type, template<typename> class GradFunctor, typename T>
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

    const int cnt = tensor_dz->shape().elem_cnt();
    size_t bytes_size = cnt * GetSizeOfDataType(tensor_dz->data_type());
    if (dptr_x) { Memset<device_type>(ctx->device_ctx(), dptr_dx, 0, bytes_size); }
    if (dptr_y) { Memset<device_type>(ctx->device_ctx(), dptr_dy, 0, bytes_size); }

    RunKernelUtil<device_type, GradFunctor, T>::BackwardKernel(ctx->device_ctx(), cnt, dptr_dz, dptr_x,
                                                     dptr_y, dptr_dx, dptr_dy);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
}  // namespace user_op
}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNEL_ELEMENTWISE_MAXIMUM_MINIMUM_KERNEL_UTIL_H