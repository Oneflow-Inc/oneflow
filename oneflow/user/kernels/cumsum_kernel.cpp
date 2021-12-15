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

    // size means dimension size, cod means coefficient of dimension
    auto size = in->shape().At(dim);
    auto space = in->shape().Count(dim);
    auto cod = in->shape().Count(dim) / size;
    auto nspace = nele / space;

    for (auto i = 0; i < nspace; i++) {
      for (auto j = 0; j < size; j++) {
        auto* tmp_pout = pout + i * cod * size + j * cod;
        auto* tmp_pin = pin + i * cod * size;
        std::copy_n(tmp_pin, cod, tmp_pout);

        for (auto k = 1; k <= j; k++) {
          auto* tmp_pin2 = tmp_pin + k * cod;
          for (auto l = 0; l < cod; l++) { tmp_pout[l] += tmp_pin2[l]; }
        }
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
