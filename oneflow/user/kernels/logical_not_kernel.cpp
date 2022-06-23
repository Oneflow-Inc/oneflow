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
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/logical_not_kernel.h"

namespace oneflow {

template<template<typename T> class UNARY_OP, typename T>
struct LogicalNotFunctor<DeviceType::kCPU, UNARY_OP, T> final {
  void operator()(ep::Stream* stream, const int64_t elem_cnt, const T* in, bool* out) {
    DoLogicalNot<UNARY_OP, T>(elem_cnt, in, out);
  }
};

template<DeviceType device_type, template<typename> class UNARY_OP, typename T, typename K>
class CpuLogicalNotKernel final : public user_op::OpKernel {
 public:
  CpuLogicalNotKernel() = default;
  ~CpuLogicalNotKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const T* x = tensor_x->dptr<T>();
    K* y = tensor_y->mut_dptr<K>();
    int64_t n = tensor_x->shape_view().elem_cnt();
    if (n != 0) { LogicalNotFunctor<device_type, UNARY_OP, T>()(ctx->stream(), n, x, y); }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_LOGICAL_NOT_KERNEL(dtype, DataType)                                      \
  REGISTER_USER_KERNEL("logical_not")                                                         \
      .SetCreateFn<CpuLogicalNotKernel<DeviceType::kCPU, UnaryFuncLogicalNot, dtype, bool>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                         \
                       && (user_op::HobDataType("x", 0) == DataType));

OF_PP_FOR_EACH_TUPLE(REGISTER_CPU_LOGICAL_NOT_KERNEL, ARITHMETIC_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ)

}  // namespace oneflow
