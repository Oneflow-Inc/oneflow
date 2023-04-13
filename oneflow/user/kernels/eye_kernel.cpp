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
#include "oneflow/user/kernels/eye_kernel_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {
namespace user_op {
template<DeviceType device_type, typename T>
class EyeKernel final : public OpKernel {
 public:
  EyeKernel() = default;
  ~EyeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    int64_t rows = ctx->Attr<int64_t>("rows");
    int64_t cols = ctx->Attr<int64_t>("cols");
    if (rows == 0 || cols == 0) { return; }
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* out = out_tensor->mut_dptr<T>();
    Memset<device_type>(
        ctx->stream(), out_tensor->mut_dptr<T>(), 0,
        out_tensor->shape_view().elem_cnt() * GetSizeOfDataType(out_tensor->data_type()));
    EyeFunctor<device_type, T>()(ctx->stream(), cols, std::min(cols, rows), out);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_EYE_KERNEL(device, dtype)                                             \
  REGISTER_USER_KERNEL("eye").SetCreateFn<EyeKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == device)                                             \
      && (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));

#define REGISTER_EYE_KERNELS_WITH_DEVICE(device) \
  REGISTER_EYE_KERNEL(device, bool)              \
  REGISTER_EYE_KERNEL(device, uint8_t)           \
  REGISTER_EYE_KERNEL(device, int8_t)            \
  REGISTER_EYE_KERNEL(device, int32_t)           \
  REGISTER_EYE_KERNEL(device, int64_t)           \
  REGISTER_EYE_KERNEL(device, float)             \
  REGISTER_EYE_KERNEL(device, double)

// Register CPU version
REGISTER_EYE_KERNELS_WITH_DEVICE(DeviceType::kCPU);

// Register CUDA version
#ifdef WITH_CUDA
REGISTER_EYE_KERNELS_WITH_DEVICE(DeviceType::kCUDA);
#endif
#undef REGISTER_EYE_KERNELS_WITH_DEVICE
#undef REGISTER_EYE_KERNEL
}  // namespace user_op
}  // namespace oneflow
