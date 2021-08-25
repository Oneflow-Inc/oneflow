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
#ifndef ONEFLOW_CORE_KERNEL_KERNEL_OBSERVER_MANAGER_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_OBSERVER_MANAGER_H_

#include "oneflow/core/kernel/kernel_observer.h"

namespace oneflow {

class KernelObserverManager final : public KernelObserver {
 public:
  KernelObserverManager();
  ~KernelObserverManager() override = default;

  void WillForward(const KernelCtx& kernel_ctx, const Kernel* kernel,
                   const std::function<Blob*(const std::string&)>& BnInOp2Blob) override;
  void DidForward(const KernelCtx& kernel_ctx, const Kernel* kernel,
                  const std::function<Blob*(const std::string&)>& BnInOp2Blob) override;

  void WillForwardShape(const KernelCtx& kernel_ctx, const Kernel* kernel,
                        const std::function<Blob*(const std::string&)>& BnInOp2Blob) override;
  void DidForwardShape(const KernelCtx& kernel_ctx, const Kernel* kernel,
                       const std::function<Blob*(const std::string&)>& BnInOp2Blob) override;

  void WillForwardDataContent(const KernelCtx& kernel_ctx, const Kernel* kernel,
                              const std::function<Blob*(const std::string&)>& BnInOp2Blob) override;
  void DidForwardDataContent(const KernelCtx& kernel_ctx, const Kernel* kernel,
                             const std::function<Blob*(const std::string&)>& BnInOp2Blob) override;

 private:
  std::vector<std::unique_ptr<KernelObserver>> kernel_observers_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_OBSERVER_MANAGER_H_
