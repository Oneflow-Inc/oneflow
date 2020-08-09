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
#include "oneflow/core/kernel/total_loss_instance_num_kernel.h"

namespace oneflow {

template<typename T>
void TotalLossInstanceNumKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& input_bns = this->op_attribute().input_bns();
  T first_val = BnInOp2Blob(input_bns.Get(0))->template dptr<T>()[0];
  for (const std::string& ibn : input_bns) {
    CHECK_EQ(BnInOp2Blob(ibn)->template dptr<T>()[0], first_val);
  }
  BnInOp2Blob("out")->template mut_dptr<T>()[0] = first_val;
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kTotalLossInstanceNumConf, TotalLossInstanceNumKernel,
                               ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
