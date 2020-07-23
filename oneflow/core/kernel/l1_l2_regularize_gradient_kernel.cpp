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
#include "oneflow/core/kernel/l1_l2_regularize_gradient_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class L1L2RegularizeGradientKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(L1L2RegularizeGradientKernel);
  L1L2RegularizeGradientKernel() = default;
  ~L1L2RegularizeGradientKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

template<DeviceType device_type, typename T>
void L1L2RegularizeGradientKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const L1L2RegularizeGradientOpConf& conf = this->op_conf().l1_l2_regularize_gradient_conf();
  const Blob* model = BnInOp2Blob("model");
  const Blob* model_diff = BnInOp2Blob("model_diff");
  Blob* out = BnInOp2Blob("out");
  L1L2RegularizeGradientKernelUtil<device_type, T>::RegularizeGradient(
      ctx.device_ctx, out->shape().elem_cnt(), model->dptr<T>(), model_diff->dptr<T>(),
      out->mut_dptr<T>(), conf.l1(), conf.l2());
}

template<DeviceType device_type, typename T>
const PbMessage& L1L2RegularizeGradientKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().l1_l2_regularize_gradient_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kL1L2RegularizeGradientConf, L1L2RegularizeGradientKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
