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
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class TotalLossInstanceNumKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TotalLossInstanceNumKernel);
  TotalLossInstanceNumKernel() = default;
  ~TotalLossInstanceNumKernel() override = default;

 private:
  void ForwardDataContent(KernelContext* ctx) const override;
};

template<typename T>
void TotalLossInstanceNumKernel<T>::ForwardDataContent(KernelContext* ctx) const {
  const auto& input_bns = this->op_attribute().input_bns();
  T first_val = ctx->BnInOp2Blob(input_bns.Get(0))->template dptr<T>()[0];
  for (const std::string& ibn : input_bns) {
    CHECK_EQ(ctx->BnInOp2Blob(ibn)->template dptr<T>()[0], first_val);
  }
  ctx->BnInOp2Blob("out")->template mut_dptr<T>()[0] = first_val;
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kTotalLossInstanceNumConf, TotalLossInstanceNumKernel,
                               ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
