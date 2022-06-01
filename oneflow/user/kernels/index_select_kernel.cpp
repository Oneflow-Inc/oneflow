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
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

template<typename T>
class IndexSelectCpuKernel final : public user_op::OpKernel {
 public:
  IndexSelectCpuKernel() = default;
  ~IndexSelectCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto& input_dim_size = x_tensor->shape().NumAxes();
    const auto& dim = ctx->Attr<int32_t>("dim");
    int64_t slice_size = 1;
    for (auto i = dim + 1; i < input_dim_size; i++) { slice_size *= x_tensor->shape().At(i); }
    const auto& input_width = slice_size * x_tensor->shape().At(dim);
    const auto& output_width = slice_size * y_tensor->shape().At(dim);
    auto outer_nums = 1;
    for (auto i = 0; i < dim; i++) { outer_nums *= x_tensor->shape().At(i); }
    auto index_size = index_tensor->shape().At(0);

    const auto* x_ptr = x_tensor->dptr<T>();
    const auto* index_ptr = index_tensor->dptr<T>();
    auto* y_ptr = y_tensor->mut_dptr<T>();
    for (auto i = 0; i < outer_nums; i++) {
      auto input_start_offset = i * input_width;
      auto output_start_offset = i * output_width;
      for (auto j = 0; j < index_size; j++) {
        int index_value = index_ptr[j];
        for (auto k = 0; k < slice_size; k++) {
          y_ptr[output_start_offset + j * slice_size + k] =
              x_ptr[input_start_offset + index_value * slice_size + k];
        }
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_INDEX_SELECT_CPU_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("index_select")                                \
      .SetCreateFn<IndexSelectCpuKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_INDEX_SELECT_CPU_KERNEL(float)
REGISTER_INDEX_SELECT_CPU_KERNEL(double)

}  // namespace oneflow
