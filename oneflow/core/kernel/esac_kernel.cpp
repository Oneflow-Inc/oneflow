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
  ctx->set_state(std::make_shared<EsacKernelState>());
}

template<typename T>
void EsacKernel<T>::ForwardDataContent(KernelContext* ctx) const {
  T value =
      static_cast<T>(CHECK_NOTNULL(dynamic_cast<EsacKernelState*>(ctx->state().get()))->value);
  *(ctx->BnInOp2Blob("out")->mut_dptr<T>()) = value;
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kEsacConf, EsacKernel, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
