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
template<typename T>
void CumSumBackward(const T* in_ptr, T* out_ptr, int64_t cs_up_space, int64_t cs_space,
                    int64_t cs_down_space, int64_t elem_cnt) {
  auto* tmp_in_ptr_base = in_ptr;
  auto* tmp_out_ptr_base = out_ptr;
  auto step = cs_space * cs_down_space;
  for (auto i = 0; i < cs_up_space; i++) {
    for (auto j = 0; j < cs_space; j++) {
      auto* tmp_in_ptr = tmp_in_ptr_base + j * cs_down_space;
      auto* tmp_out_ptr = tmp_out_ptr_base + j * cs_down_space;
      std::fill_n(tmp_out_ptr, cs_down_space, cs_space - j);
      for (auto k = 0; k < cs_down_space; k++) { tmp_out_ptr[k] *= tmp_in_ptr[k]; }
    }
    tmp_in_ptr_base += step;
    tmp_out_ptr_base += step;
  }
}

// O(n) cumprod backward, formula: cumsum(flip(dY * Y)) / X.
// Need to take care when there is at least a zero in the input.
template<typename T>
void CumProdBackward(const T* dy_ptr, T* dx_ptr, const T* output_ptr, const T* input_ptr,
                     const int64_t up_space, const int64_t space, const int64_t down_space,
                     const int64_t elem_cnt) {
  const auto step = space * down_space;
  for (size_t i = 0; i < up_space; i++) {
    // two-dims buffer for 0 elem index
    std::vector<size_t> cumsum_zeros_number(space * down_space, 0);
    auto* cumsum_zeros_number_ptr = cumsum_zeros_number.data();
    for (size_t j = 0; j < space; j++) {
      const size_t ptr_offset = j * down_space;
      auto* tmp_input_ptr = input_ptr + ptr_offset;
      auto* tmp_cumsum_zeros_number_ptr = cumsum_zeros_number_ptr + ptr_offset;
      auto* last_tmp_cumsum_zeros_number_ptr = tmp_cumsum_zeros_number_ptr - down_space;
      for (auto k = 0; k < down_space; k++) {
        int is_zero = tmp_input_ptr[k] == 0 ? 1 : 0;
        tmp_cumsum_zeros_number_ptr[k] =
            is_zero + (j == 0 ? 0 : last_tmp_cumsum_zeros_number_ptr[k]);
      }
    }
    {
      // for k < z(z is first zero index)
      std::vector<T> reverse_cumsum(down_space, 0);
      for (size_t j = 0; j < space; j++) {
        const size_t ptr_offset = (space - j - 1) * down_space;
        auto* tmp_cumsum_zeros_number_ptr = cumsum_zeros_number_ptr + ptr_offset;
        auto* tmp_dy_ptr = dy_ptr + ptr_offset;
        auto* tmp_dx_ptr = dx_ptr + ptr_offset;
        auto* tmp_output_ptr = output_ptr + ptr_offset;
        auto* tmp_input_ptr = input_ptr + ptr_offset;
        for (auto k = 0; k < down_space; k++) {
          if (tmp_cumsum_zeros_number_ptr[k] > 0) { continue; }
          reverse_cumsum[k] += tmp_output_ptr[k] * tmp_dy_ptr[k];
          tmp_dx_ptr[k] = reverse_cumsum[k] / tmp_input_ptr[k];
        }
      }
    }
    {
      // for k == z
      std::vector<size_t> first_zero(down_space, space);
      for (size_t j = 0; j < space; j++) {
        auto* tmp_cumsum_zeros_number_ptr = cumsum_zeros_number_ptr + j * down_space;
        for (size_t k = 0; k < down_space; k++) {
          if (tmp_cumsum_zeros_number_ptr[k] == 1 && first_zero[k] == space) { first_zero[k] = j; }
        }
      }
      // compute along row
      std::vector<T> cumsum_buffer(down_space, 0);
      for (size_t k = 0; k < down_space; k++) {
        auto* tmp_input_down_offset_ptr = input_ptr + k;
        auto* tmp_output_down_offset_ptr = output_ptr + k;
        auto* tmp_dy_down_offset_ptr = dy_ptr + k;
        auto* tmp_cumsum_zero_number_down_offset_ptr = cumsum_zeros_number_ptr + k;

        size_t first_zero_index = first_zero[k];
        if (first_zero_index == space) { continue; }
        auto cumprod_before_first_zero =
            first_zero_index == 0
                ? 1
                : *(tmp_output_down_offset_ptr + (first_zero_index - 1) * down_space);
        auto cumprod = 1;
        for (size_t j = first_zero_index; j < space; j++) {
          const size_t ptr_offset = j * down_space;
          auto tmp_dy = *(tmp_dy_down_offset_ptr + ptr_offset);
          auto tmp_input = *(tmp_input_down_offset_ptr + ptr_offset);
          auto tmp_cumsum_zero_number = *(tmp_cumsum_zero_number_down_offset_ptr + ptr_offset);
          if (tmp_cumsum_zero_number != 1) { continue; }
          if (j != first_zero_index) { cumprod *= tmp_input; }
          cumsum_buffer[k] += cumprod_before_first_zero * tmp_dy * cumprod;
        }
      }
      for (size_t j = 0; j < down_space; j++) {
        *(dx_ptr + first_zero[j] * down_space) = cumsum_buffer[j];
      }
    }

    input_ptr += step;
    output_ptr += step;
    dy_ptr += step;
    dx_ptr += step;
  }
}
}  // namespace

class CpuCumGradKernel : public user_op::OpKernel {
 public:
  CpuCumGradKernel() = default;
  ~CpuCumGradKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class CpuCumsumGradKernel final : public CpuCumGradKernel {
 public:
  CpuCumsumGradKernel() = default;
  ~CpuCumsumGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto elem_cnt = dy->shape().elem_cnt();
    auto dim = ctx->Attr<int64_t>("dim");
    const auto* dy_ptr = dy->dptr<T>();
    auto* dx_ptr = dx->mut_dptr<T>();

    // take cumsum's abbreviation as `cs`
    // data partition: cs_up_space|cs_space|cs_down_space
    auto cs_up_space = elem_cnt / dx->shape().Count(dim);
    auto cs_space = dx->shape().At(dim);
    auto cs_down_space = dx->shape().Count(dim + 1);

    CumSumBackward(dy_ptr, dx_ptr, cs_up_space, cs_space, cs_down_space, elem_cnt);
  }
};

#define REGISTER_CPU_CUMSUM_GRAD_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("cumsum_grad")                                 \
      .SetCreateFn<CpuCumsumGradKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CPU_CUMSUM_GRAD_KERNEL(float)
REGISTER_CPU_CUMSUM_GRAD_KERNEL(double)
#undef REGISTER_CPU_CUMSUM_GRAD_KERNEL

template<typename T>
class CpuCumProdGradKernel final : public CpuCumGradKernel {
 public:
  CpuCumProdGradKernel() = default;
  ~CpuCumProdGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    const auto* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t elem_cnt = dy->shape().elem_cnt();
    if (elem_cnt == 0) { return; }

    const auto* output_ptr = output->dptr<T>();
    const auto* input_ptr = input->dptr<T>();
    const auto* dy_ptr = dy->dptr<T>();
    auto* dx_ptr = dx->mut_dptr<T>();

    // data partition: up_space|space|down_space
    auto dim = ctx->Attr<int64_t>("dim");
    auto up_space = elem_cnt / dx->shape().Count(dim);
    auto space = dx->shape().At(dim);
    auto down_space = dx->shape().Count(dim + 1);
    if (space == 1) {
      Memcpy<DeviceType::kCPU>(ctx->stream(), dx_ptr, dy_ptr, elem_cnt * sizeof(T));
      return;
    }
    CumProdBackward(dy_ptr, dx_ptr, output_ptr, input_ptr, up_space, space, down_space, elem_cnt);
  }
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
