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
#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::VirtualKernelInit() {
  const PbMessage& op_conf = this->GetCustomizedOpConf();
  weight_decay_ = static_cast<T>(GetValFromPbMessage<float>(op_conf, "weight_decay"));
  if (!IsWeightDecaySupported()) { CHECK_EQ(weight_decay_, static_cast<T>(0)); }
}

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int64_t* train_step_ptr = BnInOp2Blob("train_step")->dptr<int64_t>();
  const float* learning_rate_ptr = BnInOp2Blob("learning_rate")->dptr<float>();
  UpdateModel(ctx.device_ctx, weight_decay_, train_step_ptr, learning_rate_ptr, BnInOp2Blob);
}

#define INSTANTIATE_KERNEL(device_type, data_type_pair) \
  template class NormalMdUpdateKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL, DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)
#undef INSTANTIATE_KERNEL

}  // namespace oneflow
