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
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/square_sum_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SquareSumKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SquareSumKernel);
  SquareSumKernel() = default;
  ~SquareSumKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
void SquareSumKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* x = BnInOp2Blob("x");
  Blob* y = BnInOp2Blob("y");
  SquareSumKernelUtil<device_type, T>::SquareSum(ctx.device_ctx, x->shape().elem_cnt(),
                                                 x->dptr<T>(), y->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSquareSumConf, SquareSumKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
