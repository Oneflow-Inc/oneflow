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

namespace oneflow {
namespace user_op {
template<DeviceType device_type, typename T>
class RangeKernel final : public OpKernel {
 public:
  RangeKernel() = default;
  ~RangeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t start = ctx->Attr<int64_t>("start");
    const int64_t delta = ctx->Attr<int64_t>("delta");
    const int64_t range_shape = ctx->Attr<int64_t>("range_shape");
    FOR_RANGE(int64_t, i, start, range_shape) {
      // In Python, range_shape = int((limit-start)/delta)
      out_tensor->mut_dptr<T>()[i] = i * delta;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RANGE_KERNEL(device, dtype)                                               \
  REGISTER_USER_KERNEL("range").SetCreateFn<RangeKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device)                                                  \
      & (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));

REGISTER_RANGE_KERNEL(DeviceType::kCPU, int64_t)

}  // namespace user_op

}  // namespace oneflow