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
#include "oneflow/core/kernel/sync_check_kernel_observer.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cuda_device_context.h"

namespace oneflow {

void SyncCheckKernelObserver::DidForwardDataContent(
    const KernelCtx& kernel_ctx, const Kernel* kernel,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
#ifdef WITH_CUDA
  auto* cuda_device_ctx = dynamic_cast<CudaDeviceCtx*>(kernel_ctx.device_ctx);
  if (cuda_device_ctx != nullptr) {
    OF_CUDA_CHECK(cudaStreamSynchronize(cuda_device_ctx->cuda_stream()))
        << kernel->op_conf().name();
  }
#endif
}

}  // namespace oneflow
