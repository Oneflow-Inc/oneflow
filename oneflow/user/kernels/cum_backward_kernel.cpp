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

namespace oneflow {
namespace {
// CumProd backward, formula: flip(cumsum(flip(dY * Y))) / X.
template<typename T>
void CumProdBackward(const T* dy_ptr, T* dx_ptr, const T* output_ptr, const T* input_ptr,
                     const int64_t up_space, const int64_t space, const int64_t down_space,
                     const int64_t elem_cnt) {
  const auto step = space * down_space;
  for (size_t i = 0; i < up_space; i++) {
    const size_t base_ptr_offset = step * i;
    const T* input_ptr_base = input_ptr + base_ptr_offset;
    const T* output_ptr_base = output_ptr + base_ptr_offset;
    const T* dy_ptr_base = dy_ptr + base_ptr_offset;
    T* dx_ptr_base = dx_ptr + base_ptr_offset;

    // Use dx as tmp buffer for finding 0 element in the input.
    for (size_t j = 0; j < space; j++) {
      const size_t ptr_offset = j * down_space;
      auto* cur_input_ptr = input_ptr_base + ptr_offset;

      auto* cumsum_zeros_number_ptr = dx_ptr_base + ptr_offset;
      auto* last_cumsum_zeros_number_ptr = cumsum_zeros_number_ptr - down_space;
      for (size_t k = 0; k < down_space; k++) {
        int is_zero = cur_input_ptr[k] == 0 ? 1 : 0;
        cumsum_zeros_number_ptr[k] = is_zero + (j == 0 ? 0 : last_cumsum_zeros_number_ptr[k]);
      }
    }

    for (size_t j = 0; j < down_space; j++) {
      const auto* cur_output_ptr = output_ptr_base + j;
      const auto* cur_input_ptr = input_ptr_base + j;
      const auto* cur_dy_ptr = dy_ptr_base + j;
      auto* cur_dx_ptr = dx_ptr_base + j;
      const auto* cumsum_zeros_number_ptr = dx_ptr_base + j;

      size_t first_zero_index = space;
      // Find index of first zero in input.
      for (size_t k = 0; k < space; k++) {
        if (cumsum_zeros_number_ptr[k * down_space] == 1) {
          first_zero_index = k;
          break;
        }
      }
      // Suppose z is index of first zero element in input,
      // for element which index is less than z grad is computed as below:
      T reverse_cumsum = 0;
      for (size_t k = 0; k < first_zero_index; k++) {
        const size_t data_offset = (first_zero_index - k - 1) * down_space;
        reverse_cumsum += cur_output_ptr[data_offset] * cur_dy_ptr[data_offset];
        cur_dx_ptr[data_offset] = reverse_cumsum / cur_input_ptr[data_offset];
      }
      // For where index is z, its grad is computed as below:
      if (first_zero_index == space) { continue; }
      T cumprod = 1;
      T cumsum = 0;
      T cumprod_before_first_zero =
          first_zero_index == 0 ? 1 : cur_output_ptr[(first_zero_index - 1) * down_space];
      for (size_t k = first_zero_index; k < space; k++) {
        const size_t data_offset = k * down_space;
        // Recover dx_ptr default value
        if (cur_dx_ptr[data_offset] >= 1) { cur_dx_ptr[data_offset] = 0; }
        if (k != first_zero_index) { cumprod *= cur_input_ptr[data_offset]; }
        cumsum += cumprod_before_first_zero * cumprod * cur_dy_ptr[data_offset];
      }
      cur_dx_ptr[first_zero_index * down_space] = cumsum;
    }
  }
}
}  // namespace

template<typename T>
class CpuCumProdGradKernel final : public user_op::OpKernel {
 public:
  CpuCumProdGradKernel() = default;
  ~CpuCumProdGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    const auto* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t elem_cnt = dy->shape_view().elem_cnt();
    if (elem_cnt == 0) { return; }

    const auto* output_ptr = output->dptr<T>();
    const auto* input_ptr = input->dptr<T>();
    const auto* dy_ptr = dy->dptr<T>();
    auto* dx_ptr = dx->mut_dptr<T>();

    // data partition: up_space|space|down_space
    auto dim = ctx->Attr<int64_t>("dim");
    auto up_space = elem_cnt / dx->shape_view().Count(dim);
    auto space = dx->shape_view().At(dim);
    auto down_space = dx->shape_view().Count(dim + 1);
    if (space == 1) {
      Memcpy<DeviceType::kCPU>(ctx->stream(), dx_ptr, dy_ptr, elem_cnt * sizeof(T));
      return;
    }
    CumProdBackward(dy_ptr, dx_ptr, output_ptr, input_ptr, up_space, space, down_space, elem_cnt);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_CUMPROD_GRAD_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("cumprod_grad")                                \
      .SetCreateFn<CpuCumProdGradKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CPU_CUMPROD_GRAD_KERNEL(float)
REGISTER_CPU_CUMPROD_GRAD_KERNEL(double)
#undef REGISTER_CPU_CUMPROD_GRAD_KERNEL

}  // namespace oneflow
