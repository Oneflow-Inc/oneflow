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
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {

namespace {

template<template<typename> class BinaryFunc, DeviceType device_type, typename T>
class ReduceKernel final : public user_op::OpKernel {
 public:
  ReduceKernel() = default;
  ~ReduceKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input_tensor", 0);
    user_op::Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("output_tensor", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");
    const Shape& reduced_shape =
        CreateReducedShape(input_tensor->shape(), {axis.begin(), axis.end()});
    NdarrayReduce<device_type, T, BinaryFunc>::Reduce(
        ctx->device_ctx(), XpuVarNdarray<T>(reduced_shape, output_tensor->mut_dptr<T>()),
        XpuVarNdarray<const T>(input_tensor->shape(), input_tensor->dptr<T>()),
        XpuVarNdarray<T>(tmp_buffer->shape(), tmp_buffer->mut_dptr<T>()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_REDUCE_XPU_KERNEL(op_name, binary_func, device, dtype)                           \
  REGISTER_USER_KERNEL(op_name)                                                                   \
      .SetCreateFn<ReduceKernel<binary_func, device, dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                        \
                       & (user_op::HobDataType("output_tensor", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("input_tensor", 0);                    \
        return in_shape->elem_cnt() * sizeof(dtype);                                              \
      });

#define REGISTER_REDUCE_ARITHMETIC_KERNELS(device, dtype)                  \
  REGISTER_REDUCE_XPU_KERNEL("reduce_prod", BinaryFuncProd, device, dtype) \
  REGISTER_REDUCE_XPU_KERNEL("reduce_min", BinaryFuncMin, device, dtype)   \
  REGISTER_REDUCE_XPU_KERNEL("reduce_sum", BinaryFuncSum, device, dtype)   \
  REGISTER_REDUCE_XPU_KERNEL("reduce_max", BinaryFuncMax, device, dtype)

#define REGISTER_REDUCE_ARITHMETIC_KERNELS_BY_DEVICE(device) \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, float)          \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, double)         \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, int8_t)         \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, int32_t)        \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, int64_t)

REGISTER_REDUCE_ARITHMETIC_KERNELS_BY_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_REDUCE_ARITHMETIC_KERNELS_BY_DEVICE(DeviceType::kGPU)
#endif

#define REGISTER_REDUCE_LOGICAL_KERNELS(device)                           \
  REGISTER_REDUCE_XPU_KERNEL("reduce_any", BinaryFuncAny, device, int8_t) \
  REGISTER_REDUCE_XPU_KERNEL("reduce_all", BinaryFuncAll, device, int8_t)

REGISTER_REDUCE_LOGICAL_KERNELS(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_REDUCE_LOGICAL_KERNELS(DeviceType::kGPU)
REGISTER_REDUCE_XPU_KERNEL("reduce_sum", BinaryFuncSum, DeviceType::kGPU, float16)
#endif

}  // namespace oneflow
