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
#include "oneflow/core/kernel/kernel_context_observer.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/stream/stream_context.h"

namespace oneflow {

void KernelContextObserver::WillForward(KernelContext* kernel_ctx, const Kernel* kernel) {
  Global<KernelObserver>::Get()->WillForward(kernel_ctx, kernel);
  auto* provider = dynamic_cast<StreamContextProvider*>(kernel_ctx->device_ctx());
  if (provider == nullptr) { return; }
  provider->GetStreamContext()->Observer()->WillForward(kernel_ctx, kernel);
}

void KernelContextObserver::DidForward(KernelContext* kernel_ctx, const Kernel* kernel) {
  Global<KernelObserver>::Get()->DidForward(kernel_ctx, kernel);
  auto* provider = dynamic_cast<StreamContextProvider*>(kernel_ctx->device_ctx());
  if (provider == nullptr) { return; }
  provider->GetStreamContext()->Observer()->DidForward(kernel_ctx, kernel);
}

void KernelContextObserver::WillForwardHeader(KernelContext* kernel_ctx, const Kernel* kernel) {
  Global<KernelObserver>::Get()->WillForwardHeader(kernel_ctx, kernel);
  auto* provider = dynamic_cast<StreamContextProvider*>(kernel_ctx->device_ctx());
  if (provider == nullptr) { return; }
  provider->GetStreamContext()->Observer()->WillForwardHeader(kernel_ctx, kernel);
}

void KernelContextObserver::DidForwardHeader(KernelContext* kernel_ctx, const Kernel* kernel) {
  Global<KernelObserver>::Get()->DidForwardHeader(kernel_ctx, kernel);
  auto* provider = dynamic_cast<StreamContextProvider*>(kernel_ctx->device_ctx());
  if (provider == nullptr) { return; }
  provider->GetStreamContext()->Observer()->DidForwardHeader(kernel_ctx, kernel);
}

void KernelContextObserver::WillForwardDataContent(KernelContext* kernel_ctx,
                                                   const Kernel* kernel) {
  Global<KernelObserver>::Get()->WillForwardDataContent(kernel_ctx, kernel);
  auto* provider = dynamic_cast<StreamContextProvider*>(kernel_ctx->device_ctx());
  if (provider == nullptr) { return; }
  provider->GetStreamContext()->Observer()->WillForwardDataContent(kernel_ctx, kernel);
}

void KernelContextObserver::DidForwardDataContent(KernelContext* kernel_ctx, const Kernel* kernel) {
  Global<KernelObserver>::Get()->DidForwardDataContent(kernel_ctx, kernel);
  auto* provider = dynamic_cast<StreamContextProvider*>(kernel_ctx->device_ctx());
  if (provider == nullptr) { return; }
  provider->GetStreamContext()->Observer()->DidForwardDataContent(kernel_ctx, kernel);
}

}  // namespace oneflow
