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
#include "oneflow/core/ndarray/binary_func.h"
namespace oneflow {

namespace {
template<typename T, template<typename> class BinaryFunc>
void CumForward(const T* in_ptr, T* out_ptr, int64_t up_space, int64_t space, int64_t down_space,
                int64_t elem_cnt) {
  std::copy_n(in_ptr, elem_cnt, out_ptr);
  auto* tmp_out_ptr_base = out_ptr;
  auto step = space * down_space;
  for (auto i = 0; i < up_space; i++) {
    for (auto j = 1; j < space; j++) {
      auto* tmp_out_ptr = tmp_out_ptr_base + j * down_space;
      auto* last_tmp_out_ptr = tmp_out_ptr - down_space;
      for (auto k = 0; k < down_space; k++) {
        tmp_out_ptr[k] = BinaryFunc<T>::Invoke(tmp_out_ptr[k], last_tmp_out_ptr[k]);
      }
    }
    tmp_out_ptr_base += step;
  }
}
}  // namespace

template<typename T, template<typename> class BinaryFunc>
class CpuCumKernel : public user_op::OpKernel {
 public:
  CpuCumKernel() = default;
  ~CpuCumKernel() = default;

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

    // data partition: up_space|space|down_space
    auto up_space = elem_cnt / in->shape().Count(dim);
    auto space = in->shape().At(dim);
    auto down_space = in->shape().Count(dim + 1);

    CumForward<T, BinaryFunc>(in_ptr, out_ptr, up_space, space, down_space, elem_cnt);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class CpuCumSumKernel final : public CpuCumKernel<T, BinaryFuncAdd> {
 public:
  CpuCumSumKernel() = default;
  ~CpuCumSumKernel() = default;
};

#define REGISTER_CUMSUM_KERNEL(dtype)                                                   \
  REGISTER_USER_KERNEL("cumsum").SetCreateFn<CpuCumSumKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                    \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CUMSUM_KERNEL(int32_t)
REGISTER_CUMSUM_KERNEL(int64_t)
REGISTER_CUMSUM_KERNEL(float)
REGISTER_CUMSUM_KERNEL(double)
#undef REGISTER_CUMSUM_KERNEL

template<typename T>
class CpuCumProdKernel final : public CpuCumKernel<T, BinaryFuncMul> {
 public:
  CpuCumProdKernel() = default;
  ~CpuCumProdKernel() = default;
};

#define REGISTER_CUMPROD_KERNEL(dtype)                                                    \
  REGISTER_USER_KERNEL("cumprod").SetCreateFn<CpuCumProdKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                      \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CUMPROD_KERNEL(int32_t)
REGISTER_CUMPROD_KERNEL(int64_t)
REGISTER_CUMPROD_KERNEL(float)
REGISTER_CUMPROD_KERNEL(double)
#undef REGISTER_CUMPROD_KERNEL
}  // namespace oneflow
