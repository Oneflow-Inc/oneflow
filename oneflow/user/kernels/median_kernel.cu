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
#include "oneflow/user/kernels/radix_sort.cuh"

namespace oneflow {

template<typename T>
class CudaMedianKernel final : public user_op::OpKernel {
 public:
  CudaMedianKernel() = default;
  ~CudaMedianKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("output", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const int32_t instance_size = in->shape_view().elem_cnt();
    const size_t sort_tensor_buffer_bytes = GetCudaAlignedSize(instance_size * sizeof(T));
    SortKeysAscending(
        in->dptr<T>(), 1, instance_size,
        reinterpret_cast<void*>(tmp_buffer->mut_dptr<char>() + sort_tensor_buffer_bytes),
        tmp_buffer->shape_view().elem_cnt() - sort_tensor_buffer_bytes, tmp_buffer->mut_dptr<T>(),
        ctx->stream()->As<ep::CudaStream>()->cuda_stream());
    Memcpy<DeviceType::kCUDA>(ctx->stream(), out->mut_dptr<T>(),
                              tmp_buffer->mut_dptr<T>() + (instance_size - 1) / 2, sizeof(T));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_MEDIAN_KERNEL(dtype)                                                   \
  REGISTER_USER_KERNEL("median")                                                             \
      .SetCreateFn<CudaMedianKernel<dtype>>()                                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                       \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value))   \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                          \
        const Shape& in_shape = ctx->InputShape("input", 0);                                 \
        const int32_t instance_size = in_shape.elem_cnt();                                   \
        size_t sort_tmp_buffer_bytes =                                                       \
            InferTempStorageForSortKeysAscending<dtype>(1, instance_size);                   \
        size_t sort_tensor_buffer_bytes = GetCudaAlignedSize(instance_size * sizeof(dtype)); \
        return sort_tmp_buffer_bytes + sort_tensor_buffer_bytes;                             \
      });

REGISTER_CUDA_MEDIAN_KERNEL(float)
REGISTER_CUDA_MEDIAN_KERNEL(double)
REGISTER_CUDA_MEDIAN_KERNEL(int8_t)
REGISTER_CUDA_MEDIAN_KERNEL(uint8_t)
REGISTER_CUDA_MEDIAN_KERNEL(int32_t)
REGISTER_CUDA_MEDIAN_KERNEL(int64_t)

}  // namespace oneflow
