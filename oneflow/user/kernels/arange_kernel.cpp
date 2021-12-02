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
#include "oneflow/user/kernels/arange_kernel_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {
namespace user_op {
template<DeviceType device_type, typename T>
class ArangeKernel final : public OpKernel {
 public:
  ArangeKernel() = default;
  ~ArangeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* output = out->mut_dptr<T>();
    const DataType dtype = ctx->Attr<DataType>("dtype");
    int64_t arange_elem_cnt = 0;
    T start = 0;
    T delta = 0;
    T limit = 0;
    if (IsIntegralDataType(dtype)) {
      start = ctx->Attr<int64_t>("integer_start");
      delta = ctx->Attr<int64_t>("integer_delta");
      limit = ctx->Attr<int64_t>("integer_limit");
      arange_elem_cnt = std::ceil(static_cast<double>(limit - start) / delta);
    } else {
      // If we use static_cast<T>(start, delta, limit) and std::ceil to calculate arange_elem_cnt,
      // it will cause rounding error.
      double float_start = ctx->Attr<double>("float_start");
      double float_delta = ctx->Attr<double>("float_delta");
      double float_limit = ctx->Attr<double>("float_limit");
      arange_elem_cnt = std::ceil(static_cast<double>(float_limit - float_start) / float_delta);
      start = static_cast<T>(float_start);
      delta = static_cast<T>(float_delta);
      limit = static_cast<T>(float_limit);
    }
    ArangeFunctor<device_type, T>()(ctx->stream(), start, delta, arange_elem_cnt, output);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ARANGE_KERNEL(device, dtype)                                                \
  REGISTER_USER_KERNEL("arange").SetCreateFn<ArangeKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == device)                                                   \
      && (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));

#define REGISTER_ARANGE_KERNELS_WITH_DEVICE(device) \
  REGISTER_ARANGE_KERNEL(device, uint8_t)           \
  REGISTER_ARANGE_KERNEL(device, int8_t)            \
  REGISTER_ARANGE_KERNEL(device, int32_t)           \
  REGISTER_ARANGE_KERNEL(device, int64_t)           \
  REGISTER_ARANGE_KERNEL(device, float)             \
  REGISTER_ARANGE_KERNEL(device, double)

// Register CPU version
REGISTER_ARANGE_KERNELS_WITH_DEVICE(DeviceType::kCPU);

// Register GPU version
#ifdef WITH_CUDA
REGISTER_ARANGE_KERNELS_WITH_DEVICE(DeviceType::kCUDA);
#endif
}  // namespace user_op
}  // namespace oneflow
