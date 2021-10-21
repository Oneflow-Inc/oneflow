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

template<DeviceType device_type, typename T>
class BroadcastLikeKernel final : public user_op::OpKernel {
 public:
  BroadcastLikeKernel() = default;
  ~BroadcastLikeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* like_tensor = ctx->Tensor4ArgNameAndIndex("like", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto& axis = ctx->Attr<std::vector<int32_t>>("broadcast_axes");
    const Shape& reduced_shape =
        CreateReducedShapeOrOnesShape(like_tensor->shape(), {axis.begin(), axis.end()});
    NdarrayUtil<device_type, T>::BroadcastTo(
        ctx->device_ctx(), XpuVarNdarray<T>(out_tensor->shape(), out_tensor->mut_dptr<T>()),
        XpuVarNdarray<const T>(reduced_shape, in_tensor->dptr<T>()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_BROADCAST_LIKE_XPU_KERNEL(device, dtype)  \
  REGISTER_USER_KERNEL("broadcast_like")                   \
      .SetCreateFn<BroadcastLikeKernel<device, dtype>>()   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#ifdef WITH_CUDA
#define REGISTER_BROADCAST_LIKE_KERNEL(dtype)                 \
  REGISTER_BROADCAST_LIKE_XPU_KERNEL(DeviceType::kCPU, dtype) \
  REGISTER_BROADCAST_LIKE_XPU_KERNEL(DeviceType::kGPU, dtype)
#else
#define REGISTER_BROADCAST_LIKE_KERNEL(dtype) \
  REGISTER_BROADCAST_LIKE_XPU_KERNEL(DeviceType::kCPU, dtype)
#endif

REGISTER_BROADCAST_LIKE_KERNEL(float)
REGISTER_BROADCAST_LIKE_KERNEL(float16)
REGISTER_BROADCAST_LIKE_KERNEL(double)
REGISTER_BROADCAST_LIKE_KERNEL(int8_t)
REGISTER_BROADCAST_LIKE_KERNEL(int32_t)
REGISTER_BROADCAST_LIKE_KERNEL(int64_t)

}  // namespace oneflow
