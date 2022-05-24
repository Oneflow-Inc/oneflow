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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/user_op_hob.h"

namespace oneflow {
namespace {

template<typename T>
T GetDtypeMatchedValue(double floating, int64_t integral);

template<>
float GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<float>(floating);
}

template<>
double GetDtypeMatchedValue(double floating, int64_t integral) {
  return floating;
}

template<>
int8_t GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<int8_t>(integral);
}

template<>
int32_t GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<int32_t>(integral);
}

template<>
int64_t GetDtypeMatchedValue(double floating, int64_t integral) {
  return integral;
}

}  // namespace

template<typename T>
class FillCpuKernel final : public user_op::OpKernel {
 public:
  FillCpuKernel() = default;
  ~FillCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    double floating_value = ctx->Attr<double>("floating_value");
    int64_t integral_value = ctx->Attr<int64_t>("integral_value");
    const T value_ = GetDtypeMatchedValue<T>(floating_value, integral_value);
    const int32_t elem_cnt = x->shape().elem_cnt();
    T* y_ptr = y->mut_dptr<T>();
    FOR_RANGE(int32_t, i, 0, elem_cnt) { y_ptr[i] = value_; }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FILL_CPU_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("fill_").SetCreateFn<FillCpuKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                 \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_FILL_CPU_KERNEL(float)
REGISTER_FILL_CPU_KERNEL(double)
REGISTER_FILL_CPU_KERNEL(int8_t)
REGISTER_FILL_CPU_KERNEL(int32_t)
REGISTER_FILL_CPU_KERNEL(int64_t)

}  // namespace oneflow
