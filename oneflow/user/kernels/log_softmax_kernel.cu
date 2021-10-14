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
#include "oneflow/core/cuda/softmax.cuh"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class LogSoftmaxKernel final : public user_op::OpKernel {
 public:
  LogSoftmaxKernel() = default;
  ~LogSoftmaxKernel() override = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    const int64_t num_classes = in->shape().At(in->shape().NumAxes() - 1);
    const int64_t num_instances = in->shape().Count(0, in->shape().NumAxes() - 1);

    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
    cuda::softmax::DirectLoad<T, ComputeType> load(in->dptr<T>(), num_classes);
    cuda::softmax::DirectStore<ComputeType, T> store(prob->mut_dptr<T>(), num_classes);
    cuda::softmax::DispatchLogSoftmax<decltype(load), decltype(store), ComputeType>(
        ctx->device_ctx()->cuda_stream(), load, store, num_instances, num_classes);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class LogSoftmaxGradKernel final : public user_op::OpKernel {
 public:
  LogSoftmaxGradKernel() = default;
  ~LogSoftmaxGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const int64_t num_classes = prob->shape().At(prob->shape().NumAxes() - 1);
    const int64_t num_instances = prob->shape().elem_cnt() / num_classes;

    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
    cuda::softmax::DirectLoad<T, ComputeType> load_prob(prob->dptr<T>(), num_classes);
    cuda::softmax::DirectLoad<T, ComputeType> load_dy(dy->dptr<T>(), num_classes);
    cuda::softmax::DirectStore<ComputeType, T> store(dx->mut_dptr<T>(), num_classes);

    cuda::softmax::DispatchLogSoftmaxGrad<decltype(load_prob), decltype(load_dy), decltype(store),
                                          ComputeType>(ctx->device_ctx()->cuda_stream(), load_prob,
                                                       load_dy, store, num_instances, num_classes);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_LOGSOFTMAX_KERNEL(device, dtype)          \
  REGISTER_USER_KERNEL("log_softmax")                      \
      .SetCreateFn<LogSoftmaxKernel<device, dtype>>()      \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobDataType("prob", 0) == GetDataType<dtype>::value));

REGISTER_LOGSOFTMAX_KERNEL(DeviceType::kGPU, half)
REGISTER_LOGSOFTMAX_KERNEL(DeviceType::kGPU, float)
REGISTER_LOGSOFTMAX_KERNEL(DeviceType::kGPU, double)

#define REGISTER_LOGSOFTMAX_GRAD_KERNEL(device, dtype)     \
  REGISTER_USER_KERNEL("log_softmax_grad")                 \
      .SetCreateFn<LogSoftmaxGradKernel<device, dtype>>()  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_LOGSOFTMAX_GRAD_KERNEL(DeviceType::kGPU, half)
REGISTER_LOGSOFTMAX_GRAD_KERNEL(DeviceType::kGPU, float)
REGISTER_LOGSOFTMAX_GRAD_KERNEL(DeviceType::kGPU, double)

}  // namespace oneflow
