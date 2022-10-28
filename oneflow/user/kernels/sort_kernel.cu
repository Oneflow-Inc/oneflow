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
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

template<typename T>
class GpuSortKernel final : public user_op::OpKernel {
 public:
  GpuSortKernel() = default;
  ~GpuSortKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    Memcpy<DeviceType::kCUDA>(ctx->stream(), out->mut_dptr<T>(), in->dptr<T>(),
                              in->shape_view().elem_cnt() * sizeof(T));
    const int32_t instance_size = in->shape_view().At(in->shape_view().NumAxes() - 1);
    const int32_t instance_num = in->shape_view().elem_cnt() / instance_size;
    const std::string& direction = ctx->Attr<std::string>("direction");
    if (direction == "ASCENDING") {
      SortKeysAscending(in->dptr<T>(), instance_num, instance_size, tmp_buffer->mut_dptr<void>(),
                        tmp_buffer->shape_view().elem_cnt(), out->mut_dptr<T>(),
                        ctx->stream()->As<ep::CudaStream>()->cuda_stream());
    } else if (direction == "DESCENDING") {
      SortKeysDescending(in->dptr<T>(), instance_num, instance_size, tmp_buffer->mut_dptr<void>(),
                         tmp_buffer->shape_view().elem_cnt(), out->mut_dptr<T>(),
                         ctx->stream()->As<ep::CudaStream>()->cuda_stream());
    } else {
      UNIMPLEMENTED();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_SORT_KERNEL(dtype)                                                    \
  REGISTER_USER_KERNEL("sort")                                                              \
      .SetCreateFn<GpuSortKernel<dtype>>()                                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                      \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))    \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                         \
        const Shape& in_shape = ctx->InputShape("in", 0);                                   \
        const int32_t instance_size = in_shape.dim_vec().back();                            \
        const int32_t instance_num = in_shape.elem_cnt() / instance_size;                   \
        const std::string& direction = ctx->Attr<std::string>("direction");                 \
        if (direction == "ASCENDING") {                                                     \
          return InferTempStorageForSortKeysAscending<dtype>(instance_num, instance_size);  \
        } else if (direction == "DESCENDING") {                                             \
          return InferTempStorageForSortKeysDescending<dtype>(instance_num, instance_size); \
        } else {                                                                            \
          UNIMPLEMENTED();                                                                  \
          return 0;                                                                         \
        }                                                                                   \
      });

REGISTER_CUDA_SORT_KERNEL(float)
REGISTER_CUDA_SORT_KERNEL(double)
REGISTER_CUDA_SORT_KERNEL(int32_t)
REGISTER_CUDA_SORT_KERNEL(int64_t)

}  // namespace oneflow
