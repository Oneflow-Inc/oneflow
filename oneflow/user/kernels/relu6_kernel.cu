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
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/cuda/elementwise.cuh"
namespace oneflow {

namespace user_op {

template<typename T>
struct Relu6Functor {
  OF_DEVICE_FUNC T operator()(T x) const {
    if (x <= static_cast<T>(0))
      return static_cast<T>(0);
    else if (x >= static_cast<T>(6))
      return static_cast<T>(6);
    else
      return x;
  }
};

template<typename T>
struct Relu6GradFunctor {
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    return (x > static_cast<T>(0) && x < static_cast<T>(6)) ? dy : static_cast<T>(0);
  }
};

template<DeviceType device_type, typename T>
class GpuRelu6Kernel final : public OpKernel {
 public:
  GpuRelu6Kernel() = default;
  ~GpuRelu6Kernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();
    const int32_t elem_cnt = in_tensor->shape().elem_cnt();
    OF_CUDA_CHECK((oneflow::cuda::elementwise::Unary(Relu6Functor<T>(), elem_cnt, out_ptr, in_ptr,
                                                     ctx->device_ctx()->cuda_stream())));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_RELU6_KERNEL(device, dtype)                                          \
  REGISTER_USER_KERNEL("relu6")                                                           \
      .SetCreateFn<GpuRelu6Kernel<device, dtype>>()                                       \
      .SetIsMatchedHob((HobDeviceTag() == device)                                         \
                       & (HobDataType("out", 0) == GetDataType<dtype>::value))            \
      .SetInplaceProposalFn(                                                              \
          [](const InferContext&, AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));             \
            return Maybe<void>::Ok();                                                     \
          });

// Register Forward GPU Kernel
REGISTER_GPU_RELU6_KERNEL(DeviceType::kGPU, float)
REGISTER_GPU_RELU6_KERNEL(DeviceType::kGPU, double)
REGISTER_GPU_RELU6_KERNEL(DeviceType::kGPU, half)

template<DeviceType device_type, typename T>
class GpuRelu6GradKernel final : public OpKernel {
 public:
  GpuRelu6GradKernel() = default;
  ~GpuRelu6GradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const T* y_ptr = y_tensor->dptr<T>();
    const T* dy_ptr = dy_tensor->dptr<T>();
    T* dx_ptr = dx_tensor->mut_dptr<T>();

    const int32_t elem_cnt = y_tensor->shape().elem_cnt();
    OF_CUDA_CHECK((oneflow::cuda::elementwise::Binary(
        Relu6GradFunctor<T>(), elem_cnt, dx_ptr, y_ptr, dy_ptr, ctx->device_ctx()->cuda_stream())));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_RELU6_BACKWARD_KERNEL(device, dtype)                                 \
  REGISTER_USER_KERNEL("relu6_grad")                                                      \
      .SetCreateFn<GpuRelu6GradKernel<device, dtype>>()                                   \
      .SetIsMatchedHob((HobDeviceTag() == device)                                         \
                       & (HobDataType("dx", 0) == GetDataType<dtype>::value))             \
      .SetInplaceProposalFn(                                                              \
          [](const InferContext&, AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));              \
            return Maybe<void>::Ok();                                                     \
          });

// Register Backward GPU Kernel
REGISTER_GPU_RELU6_BACKWARD_KERNEL(DeviceType::kGPU, double)
REGISTER_GPU_RELU6_BACKWARD_KERNEL(DeviceType::kGPU, float)
REGISTER_GPU_RELU6_BACKWARD_KERNEL(DeviceType::kGPU, half)

}  // namespace user_op

}  // namespace oneflow