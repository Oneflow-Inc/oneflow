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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/op_kernel_wrapper.h"
#include "oneflow/user/kernels/random_seed_util.h"
#include "oneflow/user/kernels/convert_memory_format_util.h"
#include "oneflow/core/ep/include/primitive/permute.h"

namespace oneflow {

template<typename Context>
std::unique_ptr<ep::primitive::Permute> NewPermutePrimitive(Context* ctx, const int& num_dims) {
  return ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(ctx->device_type(), num_dims);
}

template<typename T>
class ConvertMemoryFormatKernel final : public user_op::OpKernel {
 public:
  ConvertMemoryFormatKernel() = default;
  ~ConvertMemoryFormatKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    ConvertMemoryFormat(ctx->stream(), in->shape_view().NumAxes(), in->shape_view().data(),
                        in->data_type(), in->dptr(), out->mut_dptr(), in->memory_format(),
                        out->memory_format());
    // auto transpose = NewPermutePrimitive(ctx, in->shape_view().NumAxes());
    // CHECK(transpose);
    // std::vector<int> permutation{0, 2, 3, 1};
    // transpose->Launch(ctx->stream(), out->data_type(), out->shape_view().NumAxes(),
    //                   in->shape_view().data(), in->dptr<T>(), permutation.data(),
    //                   out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CONVERT_MEMORY_FORMAT_KERNEL(dtype)                  \
  REGISTER_USER_KERNEL("convert_memory_format")                       \
      .SetCreateFn<ConvertMemoryFormatKernel<dtype>>()                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));
REGISTER_CONVERT_MEMORY_FORMAT_KERNEL(int32_t)

}  // namespace oneflow
