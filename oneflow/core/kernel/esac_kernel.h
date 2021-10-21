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
#ifndef ONEFLOW_CORE_KERNEL_ESAC_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ESAC_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

struct EsacKernelState : public KernelState {
  int64_t value{};
};

template<typename T>
class EsacKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EsacKernel);
  EsacKernel() = default;
  ~EsacKernel() override = default;

 private:
  void VirtualKernelInit(KernelContext* ctx) override;
  void ForwardDataContent(KernelContext* ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ESAC_KERNEL_H_
