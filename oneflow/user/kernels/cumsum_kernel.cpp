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

namespace oneflow {

namespace {
template<typename T>
void cumsum_forward(const T* in_ptr, T* out_ptr, int64_t cs_up_space, int64_t cs_space,
                    int64_t cs_down_space, int64_t elem_cnt) {
  std::copy_n(in_ptr, elem_cnt, out_ptr);
  auto* tmp_out_ptr_base = out_ptr;
  auto step = cs_space * cs_down_space;
  for (auto i = 0; i < cs_up_space; i++) {
    for (auto j = 1; j < cs_space; j++) {
      auto* tmp_out_ptr = tmp_out_ptr_base + j * cs_down_space;
      auto* last_tmp_out_ptr = tmp_out_ptr - cs_down_space;
      for (auto k = 0; k < cs_down_space; k++) { tmp_out_ptr[k] += last_tmp_out_ptr[k]; }
    }
    tmp_out_ptr_base += step;
  }
}

template<typename T>
void cumsum_backward(const T* in_ptr, T* out_ptr, int64_t cs_up_space, int64_t cs_space,
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
}  // namespace

template<typename T>
class CpuCumsumKernel final : public user_op::OpKernel {
 public:
  CpuCumsumKernel() = default;
  ~CpuCumsumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* in = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto elem_cnt = in->shape().elem_cnt();
    // judge whether tensor has 0 size dimension first
    if (!elem_cnt) { return; }

    auto* out = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto dim = ctx->Attr<int64_t>("dim");
    const auto* in_ptr = in->dptr<T>();
    auto* out_ptr = out->mut_dptr<T>();

    // take cumsum's abbreviation as `cs`
    // data partition: cs_up_space|cs_space|cs_down_space
    auto cs_up_space = elem_cnt / in->shape().Count(dim);
    auto cs_space = in->shape().At(dim);
    auto cs_down_space = in->shape().Count(dim + 1);

    cumsum_forward<T>(in_ptr, out_ptr, cs_up_space, cs_space, cs_down_space, elem_cnt);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUMSUM_KERNEL(dtype)                                                   \
  REGISTER_USER_KERNEL("cumsum").SetCreateFn<CpuCumsumKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                    \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CUMSUM_KERNEL(int64_t)
REGISTER_CUMSUM_KERNEL(float)
REGISTER_CUMSUM_KERNEL(double)

template<typename T>
class CpuCumsumGradKernel final : public user_op::OpKernel {
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

    // data partition: cs_up_space|cs_space|cs_down_space
    auto cs_up_space = elem_cnt / dx->shape().Count(dim);
    auto cs_space = dx->shape().At(dim);
    auto cs_down_space = dx->shape().Count(dim + 1);

    cumsum_backward(dy_ptr, dx_ptr, cs_up_space, cs_space, cs_down_space, elem_cnt);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_CUMSUM_GRAD_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("cumsum_grad")                                 \
      .SetCreateFn<CpuCumsumGradKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CPU_CUMSUM_GRAD_KERNEL(float)
REGISTER_CPU_CUMSUM_GRAD_KERNEL(double)

}  // namespace oneflow
