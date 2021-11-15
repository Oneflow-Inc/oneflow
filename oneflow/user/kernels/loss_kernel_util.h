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
#ifndef ONEFLOW_USER_KERNELS_LOSS_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_LOSS_KERNEL_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace user_op {
namespace loss {

enum class ReductionType { kNone, kSum, kMean, kNotImplemented };

inline ReductionType GetReductionType(const std::string& reduction) {
  if (reduction == "mean") {
    return ReductionType::kMean;
  } else if (reduction == "none") {
    return ReductionType::kNone;
  } else if (reduction == "sum") {
    return ReductionType::kSum;
  }
  UNIMPLEMENTED();
  return ReductionType::kNotImplemented;
}

#define RETURN_VOID_IF_CPU(x) typename std::enable_if_t<x == DeviceType::kCPU, void>
#define RETURN_VOID_IF_GPU(x) typename std::enable_if_t<x == DeviceType::kGPU, void>

template<DeviceType device_type, typename T>
RETURN_VOID_IF_CPU(device_type)
ApplyLossReductionIfNeed(DeviceCtx* ctx, int64_t elem_cnt, const T* tmp_out, T* out,
                         const ReductionType reduction_type);

template<DeviceType device_type, typename T>
RETURN_VOID_IF_GPU(device_type)
ApplyLossReductionIfNeed(DeviceCtx* ctx, int64_t elem_cnt, const T* tmp_out, T* out,
                         const ReductionType reduction_type);

template<typename T>
user_op::InferTmpSizeFn GenDefaultInferTmpSizeFn(const std::string& input_name = "input",
                                                 const std::string& reduction_name = "reduction") {
  return [=](user_op::InferContext* ctx) {
    const int64_t n = ctx->InputShape(input_name, 0).elem_cnt();
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>(reduction_name));

    if (reduction != ReductionType::kNone) { return GetCudaAlignedSize(n * sizeof(T)); }
    return static_cast<size_t>(0);
  };
}

template<DeviceType device_type, typename T, typename R>
class SimpleLossKernel : public user_op::OpKernel {
 public:
  SimpleLossKernel() = default;
  ~SimpleLossKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto* tmp_buffer_blob = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));

    const int64_t elem_cnt = input_blob->shape().elem_cnt();

    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* out = out_blob->mut_dptr<T>();
    T* tmp_buffer = reduction != ReductionType::kNone ? tmp_buffer_blob->mut_dptr<T>() : nullptr;
    T* tmp_out = tmp_buffer;

    static_cast<const R*>(this)->ComputeOut(ctx, elem_cnt, input, target,
                                            reduction == ReductionType::kNone ? out : tmp_out);
    ApplyLossReductionIfNeed<device_type, T>(ctx->device_ctx(), elem_cnt, tmp_out, out, reduction);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T, typename R>
class SimpleLossGradKernel : public user_op::OpKernel {
 public:
  SimpleLossGradKernel() = default;
  ~SimpleLossGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    const auto* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));

    const int64_t elem_cnt = input_blob->shape().elem_cnt();

    const T* dy = dy_blob->dptr<T>();
    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* dx = dx_blob->mut_dptr<T>();

    static_cast<const R*>(this)->ComputeOut(ctx, elem_cnt, input, target, dy, dx, reduction);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
namespace {

#define REGISTER_SIMPLE_LOSS_KERNEL(name, kernel, device, dtype)                          \
  REGISTER_USER_KERNEL(name)                                                              \
      .SetCreateFn<kernel<dtype>>()                                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                               \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       & (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))   \
      .SetInferTmpSizeFn(loss::GenDefaultInferTmpSizeFn<dtype>());

#define REGISTER_SIMPLE_LOSS_GRAD_KERNEL(name, kernel, device, dtype)      \
  REGISTER_USER_KERNEL(name).SetCreateFn<kernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == device)                                 \
      & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)    \
      & (user_op::HobDataType("target", 0) == GetDataType<dtype>::value)   \
      & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)       \
      & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

}  // namespace

#define REGISTER_SIMPLE_LOSS_KERNEL_CPU(name, kernel)                \
  REGISTER_SIMPLE_LOSS_KERNEL(name, kernel, DeviceType::kCPU, float) \
  REGISTER_SIMPLE_LOSS_KERNEL(name, kernel, DeviceType::kCPU, double)

#define REGISTER_SIMPLE_LOSS_KERNEL_GPU(name, kernel)                \
  REGISTER_SIMPLE_LOSS_KERNEL(name, kernel, DeviceType::kGPU, half)  \
  REGISTER_SIMPLE_LOSS_KERNEL(name, kernel, DeviceType::kGPU, float) \
  REGISTER_SIMPLE_LOSS_KERNEL(name, kernel, DeviceType::kGPU, double)

#define REGISTER_SIMPLE_LOSS_GRAD_KERNEL_CPU(name, kernel)                \
  REGISTER_SIMPLE_LOSS_GRAD_KERNEL(name, kernel, DeviceType::kCPU, float) \
  REGISTER_SIMPLE_LOSS_GRAD_KERNEL(name, kernel, DeviceType::kCPU, double)

#define REGISTER_SIMPLE_LOSS_GRAD_KERNEL_GPU(name, kernel)                \
  REGISTER_SIMPLE_LOSS_GRAD_KERNEL(name, kernel, DeviceType::kGPU, half)  \
  REGISTER_SIMPLE_LOSS_GRAD_KERNEL(name, kernel, DeviceType::kGPU, float) \
  REGISTER_SIMPLE_LOSS_GRAD_KERNEL(name, kernel, DeviceType::kGPU, double)

}  // namespace loss
}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_LOSS_KERNEL_UTIL_H_
