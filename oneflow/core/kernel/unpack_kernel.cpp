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
#include "oneflow/core/kernel/unpack_kernel.h"
#include "oneflow/core/kernel/pack_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
void UnpackKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t out_index = res->first;
  size_t total_unpack_num = res->second;
  PackKernelUtil<device_type>::Unpack(ctx.device_ctx, out_index, total_unpack_num,
                                      BnInOp2Blob("in"), BnInOp2Blob("out"));
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kUnpackConf, UnpackKernel);

}  // namespace oneflow
