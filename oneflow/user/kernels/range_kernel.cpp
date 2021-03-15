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
#include "oneflow/user/kernels/range_kernel_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {
namespace user_op {
template<DeviceType device_type, typename T>
class RangeKernel final : public OpKernel {
 public:
  RangeKernel() = default;
  ~RangeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* output = out->mut_dptr<T>();
    const int32_t start = ctx->Attr<int64_t>("start");
    const int32_t delta = ctx->Attr<int64_t>("delta");
    const int32_t limit = ctx->Attr<int64_t>("limit");
    const int32_t range_elem_cnt =
        (((limit - start) + delta - 1) / delta);  // Do the ceil division, ceil((limit-start)/delta)
    RangeFunctor<device_type, T>()(ctx->device_ctx(), start, delta, range_elem_cnt, output);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RANGE_KERNEL(device, dtype)                                               \
  REGISTER_USER_KERNEL("range").SetCreateFn<RangeKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device)                                                  \
      & (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));

#define REGISTER_RANGE_KERNELS_WITH_DEVICE(device) \
  REGISTER_RANGE_KERNEL(device, int32_t)           \
  REGISTER_RANGE_KERNEL(device, int64_t)           \
  REGISTER_RANGE_KERNEL(device, float)             \
  REGISTER_RANGE_KERNEL(device, double)

// Register CPU version
REGISTER_RANGE_KERNELS_WITH_DEVICE(DeviceType::kCPU);

// Register GPU version
#ifdef WITH_CUDA
REGISTER_RANGE_KERNELS_WITH_DEVICE(DeviceType::kGPU);
REGISTER_RANGE_KERNEL(DeviceType::kGPU, float16);
#endif
}  // namespace user_op
}  // namespace oneflow
