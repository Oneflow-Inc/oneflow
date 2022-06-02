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
#include "oneflow/core/ep/include/primitive/memset.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

template<typename T, typename IndexT>
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
    auto slice_size = 1;
    for (auto i = dim + 1; i < input_dim_size; i++) { slice_size *= x_tensor->shape().At(i); }
    const auto& input_width = slice_size * x_tensor->shape().At(dim);
    const auto& output_width = slice_size * y_tensor->shape().At(dim);
    auto outer_nums = 1;
    for (auto i = 0; i < dim; i++) { outer_nums *= x_tensor->shape().At(i); }
    auto index_size = index_tensor->shape().At(0);
    const auto* x_ptr = x_tensor->dptr<T>();
    const auto* index_ptr = index_tensor->dptr<IndexT>();
    auto* y_ptr = y_tensor->mut_dptr<T>();
    for (auto i = 0; i < outer_nums; i++) {
      auto input_start_offset = i * input_width;
      auto output_start_offset = i * output_width;
      for (auto j = 0; j < index_size; j++) {
        IndexT index_value = index_ptr[j];
        for (auto k = 0; k < slice_size; k++) {
          y_ptr[output_start_offset + j * slice_size + k] =
              x_ptr[input_start_offset + index_value * slice_size + k];
        }
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T, typename IndexT>
class IndexSelectGradCpuKernel final : public user_op::OpKernel {
 public:
  IndexSelectGradCpuKernel() = default;
  ~IndexSelectGradCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto* dx_ptr = dx_tensor->mut_dptr<T>();
    const auto* index_ptr = index_tensor->dptr<IndexT>();
    const auto* dy_ptr = dy_tensor->dptr<T>();

    const auto& input_dim = dy_tensor->shape();
    const auto& input_dim_size = input_dim.NumAxes();
    const auto& dim = ctx->Attr<int32_t>("dim");
    const auto& output_dim = dx_tensor->shape();
    std::unique_ptr<ep::primitive::Memset> memset_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->device_type());
    CHECK(memset_primitive);
    memset_primitive->Launch(ctx->stream(), dx_ptr, 0, dx_tensor->shape().Count(0) * sizeof(T));

    auto slice_size = 1;
    for (auto i = dim + 1; i < input_dim_size; i++) { slice_size *= input_dim.At(i); }
    const auto& input_width = slice_size * input_dim.At(dim);
    const auto& output_width = slice_size * output_dim.At(dim);
    auto outer_nums = 1;
    for (auto i = 0; i < dim; i++) { outer_nums *= input_dim.At(i); }
    auto index_size = index_tensor->shape().At(0);

    for (auto i = 0; i < outer_nums; i++) {
      auto input_start_offset = i * input_width;
      auto output_start_offset = i * output_width;
      for (auto j = 0; j < index_size; j++) {
        IndexT index_value = index_ptr[j];
        for (auto k = 0; k < slice_size; k++) {
          dx_ptr[output_start_offset + index_value * slice_size + k] +=
              dy_ptr[input_start_offset + j * slice_size + k];
        }
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_INDEX_SELECT_CPU_KERNEL(dtype, index_dtype)                                      \
  REGISTER_USER_KERNEL("index_select")                                                            \
      .SetCreateFn<IndexSelectCpuKernel<dtype, index_dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                             \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)             \
                       && (user_op::HobDataType("index", 0) == GetDataType<index_dtype>::value)); \
  REGISTER_USER_KERNEL("index_select_grad")                                                       \
      .SetCreateFn<IndexSelectGradCpuKernel<dtype, index_dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                             \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)            \
                       && (user_op::HobDataType("index", 0) == GetDataType<index_dtype>::value));

REGISTER_INDEX_SELECT_CPU_KERNEL(bool, int32_t)
REGISTER_INDEX_SELECT_CPU_KERNEL(float, int32_t)
REGISTER_INDEX_SELECT_CPU_KERNEL(double, int32_t)
REGISTER_INDEX_SELECT_CPU_KERNEL(int32_t, int32_t)
REGISTER_INDEX_SELECT_CPU_KERNEL(int64_t, int32_t)
REGISTER_INDEX_SELECT_CPU_KERNEL(bool, int64_t)
REGISTER_INDEX_SELECT_CPU_KERNEL(float, int64_t)
REGISTER_INDEX_SELECT_CPU_KERNEL(double, int64_t)
REGISTER_INDEX_SELECT_CPU_KERNEL(int32_t, int64_t)
REGISTER_INDEX_SELECT_CPU_KERNEL(int64_t, int64_t)

}  // namespace oneflow
