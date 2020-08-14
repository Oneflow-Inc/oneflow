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

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class SoftmaxKernel final : public user_op::OpKernel {
 public:
  SoftmaxKernel() = default;
  ~SoftmaxKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t num_classes = x->shape().At(x->shape().NumAxes() - 1);
    const int64_t num_instances = x->shape().elem_cnt() / num_classes;

    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const size_t temp_storage_bytes = x->shape().elem_cnt() * sizeof(T);
    const size_t tmp_bytes = GetCudaAlignedSize(temp_storage_bytes / num_classes);

    T* tmp_ptr = tmp_buffer->mut_dptr<T>();
    void* temp_storage_ptr = reinterpret_cast<void*>(tmp_ptr + tmp_bytes / sizeof(T));
    SoftmaxKernelUtil<device_type, T>::ComputeProb(ctx->device_ctx(), num_instances, num_classes,
                                                   x->dptr<T>(), tmp_ptr, y->mut_dptr<T>(),
                                                   temp_storage_ptr, temp_storage_bytes);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
user_op::InferTmpSizeFn GenInferTmpSizeFn(const std::string& bn) {
  return [bn](user_op::InferContext* ctx) {
    const Shape* x = ctx->Shape4ArgNameAndIndex(bn, 0);
    const size_t num_classes = x->dim_vec().back();
    size_t temp_storage_bytes = GetCudaAlignedSize(x->elem_cnt() * sizeof(T));           // [i][j]
    size_t tmp_or_sum_vec_bytes = GetCudaAlignedSize(temp_storage_bytes / num_classes);  //[i]
    return tmp_or_sum_vec_bytes + temp_storage_bytes;
  };
}

#define REGISTER_SOFTMAX_KERNEL(device, dtype)                                          \
  REGISTER_USER_KERNEL("softmax")                                                       \
      .SetCreateFn<SoftmaxKernel<device, dtype>>()                                      \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(GenInferTmpSizeFn<dtype>("in"));

REGISTER_SOFTMAX_KERNEL(DeviceType::kCPU, float)
REGISTER_SOFTMAX_KERNEL(DeviceType::kCPU, double)
#ifdef WITH_CUDA
REGISTER_SOFTMAX_KERNEL(DeviceType::kGPU, float16)
REGISTER_SOFTMAX_KERNEL(DeviceType::kGPU, float)
REGISTER_SOFTMAX_KERNEL(DeviceType::kGPU, double)
#endif

template<DeviceType device_type, typename T>
class SoftmaxGradKernel final : public user_op::OpKernel {
 public:
  SoftmaxGradKernel() = default;
  ~SoftmaxGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const int64_t num_classes = y->shape().At(y->shape().NumAxes() - 1);
    const int64_t num_instances = y->shape().elem_cnt() / num_classes;

    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const size_t temp_storage_bytes = y->shape().elem_cnt() * sizeof(T);
    const size_t sum_vec_bytes = GetCudaAlignedSize(temp_storage_bytes / num_classes);

    T* sum_vec_ptr = tmp_buffer->mut_dptr<T>();
    void* temp_storage_ptr = reinterpret_cast<void*>(sum_vec_ptr + sum_vec_bytes / sizeof(T));
    SoftmaxKernelUtil<device_type, T>::ComputeDiff(
        ctx->device_ctx(), num_instances, num_classes, dy->dptr<T>(), y->dptr<T>(), sum_vec_ptr,
        dx->mut_dptr<T>(), temp_storage_ptr, temp_storage_bytes);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SOFTMAX_GRAD_KERNEL(device, dtype)                                    \
  REGISTER_USER_KERNEL("softmax_grad")                                                 \
      .SetCreateFn<SoftmaxGradKernel<device, dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(GenInferTmpSizeFn<dtype>("dx"));

REGISTER_SOFTMAX_GRAD_KERNEL(DeviceType::kCPU, float)
REGISTER_SOFTMAX_GRAD_KERNEL(DeviceType::kCPU, double)
#ifdef WITH_CUDA
REGISTER_SOFTMAX_GRAD_KERNEL(DeviceType::kGPU, float16)
REGISTER_SOFTMAX_GRAD_KERNEL(DeviceType::kGPU, float)
REGISTER_SOFTMAX_GRAD_KERNEL(DeviceType::kGPU, double)
#endif

}  // namespace

}  // namespace oneflow
