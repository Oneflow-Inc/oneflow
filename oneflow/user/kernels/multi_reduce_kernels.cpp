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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/user/kernels/multi_reduce_kernel_util.h"
#if defined(__CUDA_ARCH__)
#include "oneflow/user/kernels/multi_reduce_kernel_util.cuh"
#endif

namespace oneflow {
namespace user_op {

template<DeviceType device_type, typename T>
class MultiReduceSumPowAbsKernel final : public user_op::OpKernel,
                                         public user_op::CudaGraphSupport {
 public:
  MultiReduceSumPowAbsKernel() = default;
  ~MultiReduceSumPowAbsKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    std::vector<MultiReduceParam<T>> params;
    params.resize(ctx->input_size("x"));
    for (size_t i = 0; i < params.size(); ++i) {
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", i);
      params[i].size = x->shape().elem_cnt();
      params[i].data = x->dptr<T>();
    }
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    T* y_dptr = y->mut_dptr<T>();
    float p = ctx->Attr<float>("p");
    if (p == 0) {
      PowByZero<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_sum{};
      reduce_sum(ctx->stream(), func, params, 0, y_dptr);
    } else if (p == 1) {
      Abs<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_sum{};
      reduce_sum(ctx->stream(), func, params, 0, y_dptr);
    } else if (p == 2) {
      Square<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_sum{};
      reduce_sum(ctx->stream(), func, params, 0, y_dptr);
    } else {
      AbsPow<T> func{p};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_sum{};
      reduce_sum(ctx->stream(), func, params, 0, y_dptr);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MULTI_REDUCE_SUM_POW_ABS_KERNEL(device, dtype) \
  REGISTER_USER_KERNEL("multi_reduce_sum_pow_abs")              \
      .SetCreateFn<MultiReduceSumPowAbsKernel<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)     \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_MULTI_REDUCE_SUM_POW_ABS_KERNEL(DeviceType::kCPU, float)
REGISTER_MULTI_REDUCE_SUM_POW_ABS_KERNEL(DeviceType::kCPU, double)
#if defined(__CUDA_ARCH__)
REGISTER_MULTI_REDUCE_SUM_POW_ABS_KERNEL(DeviceType::kCUDA, float)
REGISTER_MULTI_REDUCE_SUM_POW_ABS_KERNEL(DeviceType::kCUDA, double)
REGISTER_MULTI_REDUCE_SUM_POW_ABS_KERNEL(DeviceType::kCUDA, half)
#endif

enum class Ximum {
  kMax = 0,
  kMin = 1,
};

template<DeviceType device_type, typename T, Ximum X>
class MultiReduceXimumAbsKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    std::vector<MultiReduceParam<T>> params;
    params.resize(ctx->input_size("x"));
    for (size_t i = 0; i < params.size(); ++i) {
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", i);
      params[i].size = x->shape().elem_cnt();
      params[i].data = x->dptr<T>();
    }
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    Abs<T> abs{};
    if (X == Ximum::kMax) {
      MultiReduce<device_type, T, decltype(abs), BinaryMax<T>> reduce_max{};
      reduce_max(ctx->stream(), abs, params, 0, y->mut_dptr<T>());
    } else if (X == Ximum::kMin) {
      MultiReduce<device_type, T, decltype(abs), BinaryMin<T>> reduce_min{};
      reduce_min(ctx->stream(), abs, params, std::numeric_limits<T>::max(), y->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MULTI_REDUCE_XIMUM_ABS_KERNEL(op_type_name, ximum_enum, device, dtype) \
  REGISTER_USER_KERNEL(op_type_name)                                                    \
      .SetCreateFn<MultiReduceXimumAbsKernel<device, dtype, ximum_enum>>()              \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#define REGISTER_MULTI_REDUCE_XIMUM_ABS_KERNELS(device, dtype)                                     \
  REGISTER_MULTI_REDUCE_XIMUM_ABS_KERNEL("multi_reduce_max_abs", Ximum::kMax, device, dtype)       \
  REGISTER_MULTI_REDUCE_XIMUM_ABS_KERNEL("multi_reduce_min_abs", Ximum::kMin, device, dtype)       \
  REGISTER_MULTI_REDUCE_XIMUM_ABS_KERNEL("local_multi_reduce_max_abs", Ximum::kMax, device, dtype) \
  REGISTER_MULTI_REDUCE_XIMUM_ABS_KERNEL("local_multi_reduce_min_abs", Ximum::kMin, device, dtype)

REGISTER_MULTI_REDUCE_XIMUM_ABS_KERNELS(DeviceType::kCPU, float)
REGISTER_MULTI_REDUCE_XIMUM_ABS_KERNELS(DeviceType::kCPU, double)
#if defined(__CUDA_ARCH__)
REGISTER_MULTI_REDUCE_XIMUM_ABS_KERNELS(DeviceType::kCUDA, float)
REGISTER_MULTI_REDUCE_XIMUM_ABS_KERNELS(DeviceType::kCUDA, double)
REGISTER_MULTI_REDUCE_XIMUM_ABS_KERNELS(DeviceType::kCUDA, half)
#endif

}  // namespace user_op
}  // namespace oneflow
