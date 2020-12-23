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
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename T>
class CpuRelu6Kernel final : public OpKernel {
 public:
  CpuRelu6Kernel() = default;
  ~CpuRelu6Kernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();

    const int32_t elem_cnt = in_tensor->shape().elem_cnt();
    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      // if(in_ptr[i] >= static_cast<T>(6)){
      //     out_ptr[i] = static_cast<T>(6);
      // }
      // else if(in_ptr[i] <= static_cast<T>(0)){
      //     out_ptr[i] = static_cast<T>(0);
      // }
      // else{
      //     out_ptr[i] = in_ptr[i];
      // }
      out_ptr[i] = std::min(std::max(static_cast<T>(0), in_ptr[i]), static_cast<T>(6));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_RELU6_KERNEL(device, dtype)                                          \
  REGISTER_USER_KERNEL("relu6")                                                           \
      .SetCreateFn<CpuRelu6Kernel<device, dtype>>()                                       \
      .SetIsMatchedHob((HobDeviceTag() == device)                                         \
                       & (HobDataType("out", 0) == GetDataType<dtype>::value))            \
      .SetInplaceProposalFn(                                                              \
          [](const InferContext&, AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));             \
            return Maybe<void>::Ok();                                                     \
          });

// Register Forward CPU Kernel
REGISTER_CPU_RELU6_KERNEL(DeviceType::kCPU, float)
REGISTER_CPU_RELU6_KERNEL(DeviceType::kCPU, double)

template<DeviceType device_type, typename T>
class CpuRelu6GradKernel final : public OpKernel {
 public:
  CpuRelu6GradKernel() = default;
  ~CpuRelu6GradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const T* y_ptr = y_tensor->dptr<T>();
    const T* dy_ptr = dy_tensor->dptr<T>();
    T* dx_ptr = dx_tensor->mut_dptr<T>();

    const int32_t elem_cnt = y_tensor->shape().elem_cnt();
    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      dx_ptr[i] = (y_ptr[i] > static_cast<T>(0) && y_ptr[i] < static_cast<T>(6))
                      ? dy_ptr[i]
                      : static_cast<T>(0);  // relu6 grad is 1*dy
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_RELU6_BACKWARD_KERNEL(device, dtype)                                 \
  REGISTER_USER_KERNEL("relu6_grad")                                                      \
      .SetCreateFn<CpuRelu6GradKernel<device, dtype>>()                                   \
      .SetIsMatchedHob((HobDeviceTag() == device)                                         \
                       & (HobDataType("dx", 0) == GetDataType<dtype>::value))             \
      .SetInplaceProposalFn(                                                              \
          [](const InferContext&, AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));              \
            return Maybe<void>::Ok();                                                     \
          });

// Register Backward CPU Kernel
REGISTER_CPU_RELU6_BACKWARD_KERNEL(DeviceType::kCPU, float)
REGISTER_CPU_RELU6_BACKWARD_KERNEL(DeviceType::kCPU, double)

}  // namespace user_op

}  // namespace oneflow