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
#include "oneflow/core/kernel/profiler_kernel_observer.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/profiler/kernel.h"

namespace oneflow {

void ProfilerKernelObserver::WillForwardDataContent(KernelContext* kernel_ctx,
                                                    const Kernel* kernel) {
  OF_PROFILER_ONLY_CODE(profiler::TraceKernelForwardDataContentStart(kernel_ctx, kernel));
}

void ProfilerKernelObserver::DidForwardDataContent(KernelContext* kernel_ctx,
                                                   const Kernel* kernel) {
  OF_PROFILER_ONLY_CODE(profiler::TraceKernelForwardDataContentEnd(kernel_ctx, kernel));
}

}  // namespace oneflow
