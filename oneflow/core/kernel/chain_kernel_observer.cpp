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
#include "oneflow/core/kernel/chain_kernel_observer.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

void ChainKernelObserver::WillForward(KernelContext* kernel_ctx, const Kernel* kernel) {
  for (const auto& observer : kernel_observers_) { observer->WillForward(kernel_ctx, kernel); }
}

void ChainKernelObserver::DidForward(KernelContext* kernel_ctx, const Kernel* kernel) {
  for (const auto& observer : kernel_observers_) { observer->DidForward(kernel_ctx, kernel); }
}

void ChainKernelObserver::WillForwardHeader(KernelContext* kernel_ctx, const Kernel* kernel) {
  for (const auto& observer : kernel_observers_) {
    observer->WillForwardHeader(kernel_ctx, kernel);
  }
}

void ChainKernelObserver::DidForwardHeader(KernelContext* kernel_ctx, const Kernel* kernel) {
  for (const auto& observer : kernel_observers_) { observer->DidForwardHeader(kernel_ctx, kernel); }
}

void ChainKernelObserver::WillForwardDataContent(KernelContext* kernel_ctx, const Kernel* kernel) {
  for (const auto& observer : kernel_observers_) {
    observer->WillForwardDataContent(kernel_ctx, kernel);
  }
}

void ChainKernelObserver::DidForwardDataContent(KernelContext* kernel_ctx, const Kernel* kernel) {
  for (const auto& observer : kernel_observers_) {
    observer->DidForwardDataContent(kernel_ctx, kernel);
  }
}

}  // namespace oneflow
