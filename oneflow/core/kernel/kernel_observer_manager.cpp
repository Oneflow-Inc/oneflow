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
#include "oneflow/core/kernel/kernel_observer_manager.h"
#include "oneflow/core/kernel/check_numerics_kernel_observer.h"
#include "oneflow/core/kernel/sync_check_kernel_observer.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

KernelObserverManager::KernelObserverManager() {
  if (ParseBooleanFromEnv("ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS", false)) {
    LOG(WARNING) << "env ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS has been enabled, it will impact "
                    "performance";
    kernel_observers_.emplace_back(new CheckNumericsKernelObserver());
  }
  if (ParseBooleanFromEnv("ONEFLOW_DEBUG_KERNEL_SYNC_CHECK", false)) {
    LOG(WARNING)
        << "env ONEFLOW_DEBUG_KERNEL_SYNC_CHECK has been enabled, it will impact performance";
    kernel_observers_.emplace_back(new SyncCheckKernelObserver());
  }
}

void KernelObserverManager::WillForward(
    const KernelCtx& kernel_ctx, const Kernel* kernel,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  for (const auto& observer : kernel_observers_) {
    observer->WillForward(kernel_ctx, kernel, BnInOp2Blob);
  }
}

void KernelObserverManager::DidForward(
    const KernelCtx& kernel_ctx, const Kernel* kernel,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  for (const auto& observer : kernel_observers_) {
    observer->DidForward(kernel_ctx, kernel, BnInOp2Blob);
  }
}

void KernelObserverManager::WillForwardShape(
    const KernelCtx& kernel_ctx, const Kernel* kernel,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  for (const auto& observer : kernel_observers_) {
    observer->WillForwardShape(kernel_ctx, kernel, BnInOp2Blob);
  }
}

void KernelObserverManager::DidForwardShape(
    const KernelCtx& kernel_ctx, const Kernel* kernel,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  for (const auto& observer : kernel_observers_) {
    observer->DidForwardShape(kernel_ctx, kernel, BnInOp2Blob);
  }
}

void KernelObserverManager::WillForwardDataContent(
    const KernelCtx& kernel_ctx, const Kernel* kernel,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  for (const auto& observer : kernel_observers_) {
    observer->WillForwardDataContent(kernel_ctx, kernel, BnInOp2Blob);
  }
}

void KernelObserverManager::DidForwardDataContent(
    const KernelCtx& kernel_ctx, const Kernel* kernel,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  for (const auto& observer : kernel_observers_) {
    observer->DidForwardDataContent(kernel_ctx, kernel, BnInOp2Blob);
  }
}

}  // namespace oneflow
