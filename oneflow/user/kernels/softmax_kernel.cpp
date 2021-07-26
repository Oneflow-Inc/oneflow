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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/user/kernels/softmax_kernel_util.h"
#include "oneflow/core/rocm/softmax_rocm.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class SoftmaxKernel final : public user_op::OpKernel {
 public:
  SoftmaxKernel() = default;
  ~SoftmaxKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t num_classes = in->shape().At(in->shape().NumAxes() - 1);
    const int64_t num_instances = in->shape().Count(0, in->shape().NumAxes() - 1);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const size_t temp_storage_bytes = tmp_buffer->shape().elem_cnt();
    SoftmaxKernelUtil<device_type, T>::ComputeProb(ctx->device_ctx(), num_instances, num_classes,
                                                   in->dptr<T>(), out->mut_dptr<T>(),
                                                   tmp_buffer->mut_dptr(), temp_storage_bytes);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
user_op::InferTmpSizeFn GenFwInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const Shape& in_shape = ctx->InputShape("in", 0);
    const int64_t num_classes = in_shape.At(in_shape.NumAxes() - 1);
    const int64_t num_instances = in_shape.Count(0, in_shape.NumAxes() - 1);
    return SoftmaxKernelUtil<device_type, T>::GetComputeProbTempStorageSizeInBytes(num_instances,
                                                                                   num_classes);
  };
}

#define REGISTER_SOFTMAX_KERNEL(device, dtype)                                          \
  REGISTER_USER_KERNEL("softmax")                                                       \
      .SetCreateFn<SoftmaxKernel<device, dtype>>()                                      \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(GenFwInferTmpSizeFn<device, dtype>());

REGISTER_SOFTMAX_KERNEL(DeviceType::kCPU, float)
REGISTER_SOFTMAX_KERNEL(DeviceType::kCPU, double)

template<DeviceType device_type, typename T>
class SoftmaxGradKernel final : public user_op::OpKernel {
 public:
  SoftmaxGradKernel() = default;
  ~SoftmaxGradKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const int64_t num_classes = y->shape().At(y->shape().NumAxes() - 1);
    const int64_t num_instances = y->shape().elem_cnt() / num_classes;

    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const size_t temp_storage_bytes = tmp_buffer->shape().elem_cnt();

    SoftmaxKernelUtil<device_type, T>::ComputeDiff(ctx->device_ctx(), num_instances, num_classes,
                                                   dy->dptr<T>(), y->dptr<T>(), dx->mut_dptr<T>(),
                                                   tmp_buffer->mut_dptr(), temp_storage_bytes);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
user_op::InferTmpSizeFn GenBwInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const Shape& dy_shape = ctx->InputShape("dy", 0);
    const int64_t num_classes = dy_shape.At(dy_shape.NumAxes() - 1);
    const int64_t num_instances = dy_shape.Count(0, dy_shape.NumAxes() - 1);
    return SoftmaxKernelUtil<device_type, T>::GetComputeDiffTempStorageSizeInBytes(num_instances,
                                                                                   num_classes);
  };
}

#define REGISTER_SOFTMAX_GRAD_KERNEL(device, dtype)                                    \
  REGISTER_USER_KERNEL("softmax_grad")                                                 \
      .SetCreateFn<SoftmaxGradKernel<device, dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(GenBwInferTmpSizeFn<device, dtype>());

REGISTER_SOFTMAX_GRAD_KERNEL(DeviceType::kCPU, float)
REGISTER_SOFTMAX_GRAD_KERNEL(DeviceType::kCPU, double)

}  // namespace

#if defined(WITH_HIP)

template<typename T>
class SoftmaxGpuKernel final : public user_op::OpKernel {
 public:
  SoftmaxGpuKernel() = default;
  ~SoftmaxGpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& in_shape = in->shape();
    const int64_t cols = in_shape.At(in_shape.NumAxes() - 1);
    const int64_t rows = in_shape.Count(0, in_shape.NumAxes() - 1);
    using ComputeType = typename rocm::softmax::DefaultComputeType<T>::type;
    rocm::softmax::DirectLoad<T, ComputeType> load(in->dptr<T>(), cols);
    rocm::softmax::DirectStore<ComputeType, T> store(out->mut_dptr<T>(), cols);
    rocm::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
        ctx->device_ctx()->rocm_stream(), load, store, rows, cols);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SOFTMAX_GPU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("softmax").SetCreateFn<SoftmaxGpuKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == DeviceType::kGPU)                                    \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

// REGISTER_SOFTMAX_GPU_KERNEL(half)
REGISTER_SOFTMAX_GPU_KERNEL(float)
REGISTER_SOFTMAX_GPU_KERNEL(double)
#undef REGISTER_SOFTMAX_GPU_KERNEL

template<typename T>
class SoftmaxGpuGradKernel final : public user_op::OpKernel {
 public:
  SoftmaxGpuGradKernel() = default;
  ~SoftmaxGpuGradKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t cols = y->shape().At(y->shape().NumAxes() - 1);
    const int64_t rows = y->shape().elem_cnt() / cols;
    using ComputeType = typename rocm::softmax::DefaultComputeType<T>::type;
    rocm::softmax::DirectLoad<T, ComputeType> load_y(y->dptr<T>(), cols);
    rocm::softmax::DirectLoad<T, ComputeType> load_dy(dy->dptr<T>(), cols);
    rocm::softmax::DirectStore<ComputeType, T> store(dx->mut_dptr<T>(), cols);
    rocm::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy), decltype(store),
                                       ComputeType>(ctx->device_ctx()->rocm_stream(), load_y,
                                                    load_dy, store, rows, cols);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SOFTMAX_GPU_GRAD_KERNEL(dtype)                          \
  REGISTER_USER_KERNEL("softmax_grad")                               \
      .SetCreateFn<SoftmaxGpuGradKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU) \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

// REGISTER_SOFTMAX_GPU_GRAD_KERNEL(half)
REGISTER_SOFTMAX_GPU_GRAD_KERNEL(float)
REGISTER_SOFTMAX_GPU_GRAD_KERNEL(double)
#undef REGISTER_SOFTMAX_GPU_GRAD_KERNEL

#endif

}  // namespace oneflow
