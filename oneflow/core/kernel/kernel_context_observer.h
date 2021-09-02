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
#ifndef ONEFLOW_CORE_KERNEL_CONTEXT_OBSERVER_H_
#define ONEFLOW_CORE_KERNEL_CONTEXT_OBSERVER_H_

#include "oneflow/core/kernel/kernel_observer.h"

namespace oneflow {

class KernelContextObserver final : public KernelObserver {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelContextObserver);
  KernelContextObserver() = default;
  ~KernelContextObserver() override = default;

  void WillForward(KernelContext* kernel_ctx, const Kernel* kernel) override;
  void DidForward(KernelContext* kernel_ctx, const Kernel* kernel) override;

  void WillForwardHeader(KernelContext* kernel_ctx, const Kernel* kernel) override;
  void DidForwardHeader(KernelContext* kernel_ctx, const Kernel* kernel) override;

  void WillForwardDataContent(KernelContext* kernel_ctx, const Kernel* kernel) override;
  void DidForwardDataContent(KernelContext* kernel_ctx, const Kernel* kernel) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONTEXT_OBSERVER_H_
