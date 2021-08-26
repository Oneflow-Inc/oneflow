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
#include "oneflow/core/kernel/blob_access_checker_kernel_observer.h"
#include "oneflow/core/kernel/profiler_kernel_observer.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

KernelObserverManager::KernelObserverManager() {
  if (ParseBooleanFromEnv("ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS", false)) {
    LOG(WARNING) << "Environment variable ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS has been set to "
                    "a truthy value, it will impact "
                    "performance";
    kernel_observers_.emplace_back(new CheckNumericsKernelObserver());
  }
  if (ParseBooleanFromEnv("ONEFLOW_DEBUG_KERNEL_SYNC_CHECK", false)) {
    LOG(WARNING) << "Environment variable ONEFLOW_DEBUG_KERNEL_SYNC_CHECK has been set to a truthy "
                    "value, it will impact performance";
    kernel_observers_.emplace_back(new SyncCheckKernelObserver());
  }
  if (!ParseBooleanFromEnv("ONEFLOW_KERNEL_DISABLE_BLOB_ACCESS_CHECKER", false)) {
    kernel_observers_.emplace_back(new BlobAccessCheckerKernelObserver());
  }
  kernel_observers_.emplace_back(new ProfilerKernelObserver());
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

void KernelObserverManager::WillForwardHeader(
    const KernelCtx& kernel_ctx, const Kernel* kernel,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  for (const auto& observer : kernel_observers_) {
    observer->WillForwardHeader(kernel_ctx, kernel, BnInOp2Blob);
  }
}

void KernelObserverManager::DidForwardHeader(
    const KernelCtx& kernel_ctx, const Kernel* kernel,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  for (const auto& observer : kernel_observers_) {
    observer->DidForwardHeader(kernel_ctx, kernel, BnInOp2Blob);
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
