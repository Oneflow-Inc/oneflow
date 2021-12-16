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

template<typename T>
class CpuCumsumKernel final : public user_op::OpKernel {
 public:
  CpuCumsumKernel() = default;
  ~CpuCumsumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // judge whether tensor has 0 size dimension first
    const auto* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    auto nele = in->shape().elem_cnt();
    if (!nele) { return; }

    auto* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto dim = ctx->Attr<int64_t>("dim");
    const auto* pin = in->dptr<T>();
    auto* pout = out->mut_dptr<T>();

    // take cumsum's abbreviation as `cs`
    // data partition: cs_up_space|cs_space|cs_down_space
    auto cs_up_space = nele / in->shape().Count(dim);
    auto cs_space = in->shape().At(dim);
    auto cs_down_space = in->shape().Count(dim) / cs_space;

    std::copy_n(pin, nele, pout);
    for (auto i = 0; i < cs_up_space; i++) {
      auto* tmp_pout_base = pout + i * cs_space * cs_down_space;
      for (auto j = 1; j < cs_space; j++) {
        auto* tmp_pout = tmp_pout_base + j * cs_down_space;
        auto* last_tmp_pout = tmp_pout - cs_down_space;
        for (auto k = 0; k < cs_down_space; k++) { tmp_pout[k] += last_tmp_pout[k]; }
      }
    }
  }

  // TODO: what's it used for?
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUMSUM_KERNEL(dtype)                                                   \
  REGISTER_USER_KERNEL("cumsum").SetCreateFn<CpuCumsumKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                    \
      && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CUMSUM_KERNEL(float)
REGISTER_CUMSUM_KERNEL(double)

}  // namespace oneflow
