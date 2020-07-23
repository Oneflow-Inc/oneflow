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
#include "oneflow/core/kernel/tuple_identity_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void TupleIdentityKernel<device_type>::ForwardHeader(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& input_bns = this->op_attribute().input_bns();
  const auto& output_bns = this->op_attribute().output_bns();
  CHECK_EQ(input_bns.size(), output_bns.size());
  FOR_RANGE(int, i, 0, input_bns.size()) {
    Blob* out_blob = BnInOp2Blob(output_bns.Get(i));
    out_blob->CopyHeaderFrom(ctx.device_ctx, BnInOp2Blob(input_bns.Get(i)));
  }
}

template<DeviceType device_type>
void TupleIdentityKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& input_bns = this->op_attribute().input_bns();
  const auto& output_bns = this->op_attribute().output_bns();
  CHECK_EQ(input_bns.size(), output_bns.size());
  FOR_RANGE(int, i, 0, input_bns.size()) {
    Blob* out_blob = BnInOp2Blob(output_bns.Get(i));
    out_blob->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob(input_bns.Get(i)));
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kTupleIdentityConf, TupleIdentityKernel);

}  // namespace oneflow
