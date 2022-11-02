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
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

template<typename Context>
std::unique_ptr<ep::primitive::Fill> NewFillPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::FillFactory>(ctx->device_type(), data_type);
}

}  // namespace

class FillKernel final : public user_op::OpKernel {
 public:
  FillKernel() = default;
  ~FillKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const bool is_floating_value = ctx->Attr<bool>("is_floating_value");
    const Scalar value = is_floating_value ? Scalar(ctx->Attr<double>("floating_value"))
                                           : Scalar(ctx->Attr<int64_t>("integral_value"));
    const int32_t elem_cnt = in->shape_view().elem_cnt();
    CHECK_GE(elem_cnt, 0);
    if (elem_cnt == 0) { return; }
    std::unique_ptr<ep::primitive::Fill> fill = NewFillPrimitive(ctx);
    CHECK(fill);
    fill->Launch(ctx->stream(), out->mut_dptr(), value, elem_cnt);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto FillPrimitiveExists() {
  return hob::make_custom("FillPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewFillPrimitive(&ctx).operator bool();
  });
}

template<typename T>
class FillTensorCpuKernel final : public user_op::OpKernel {
 public:
  FillTensorCpuKernel() = default;
  ~FillTensorCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);
    const T value_ = value->dptr<T>()[0];
    const int32_t elem_cnt = in->shape_view().elem_cnt();
    T* out_ptr = out->mut_dptr<T>();
    FOR_RANGE(int32_t, i, 0, elem_cnt) { out_ptr[i] = value_; }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FILL_CPU_KERNEL(dtype)                               \
  REGISTER_USER_KERNEL("fill_tensor_")                                \
      .SetCreateFn<FillTensorCpuKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_FILL_CPU_KERNEL(float)
REGISTER_FILL_CPU_KERNEL(float16)
REGISTER_FILL_CPU_KERNEL(double)
REGISTER_FILL_CPU_KERNEL(int8_t)
REGISTER_FILL_CPU_KERNEL(int32_t)
REGISTER_FILL_CPU_KERNEL(int64_t)
REGISTER_USER_KERNEL("fill_").SetCreateFn<FillKernel>().SetIsMatchedHob(FillPrimitiveExists()
                                                                        == true);

}  // namespace oneflow
