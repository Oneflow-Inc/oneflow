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
#include "oneflow/core/kernel/esac_kernel.h"

namespace oneflow {

template<typename T>
void EsacKernel<T>::VirtualKernelInit(KernelContext* ctx) {
  ctx->set_state(new int64_t);
}

template<typename T>
void EsacKernel<T>::DestroyState(void* state) const {
  delete static_cast<int64_t*>(state);
}

template<typename T>
void EsacKernel<T>::ForwardDataContent(KernelContext* ctx) const {
  T value = static_cast<T>(*static_cast<int64_t*>(ctx->state()));
  KernelUtil<DeviceType::kCPU, T>::Set(ctx->device_ctx(), value,
                                       ctx->BnInOp2Blob("out")->mut_dptr<T>());
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kEsacConf, EsacKernel, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
