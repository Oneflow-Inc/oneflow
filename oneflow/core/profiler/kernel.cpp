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

#include "oneflow/core/profiler/kernel.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cuda_device_context.h"

namespace oneflow {

namespace profiler {

namespace {

bool profile_cuda_memory_bandwidth = false;

COMMAND(ParseBoolFlagFromEnv("ONEFLOW_PROFILER_KERNEL_PROFILE_CUDA_MEMORY_BANDWIDTH",
                             &profile_cuda_memory_bandwidth));

thread_local cudaEvent_t cuda_memory_bandwidth_profile_start_event = nullptr;
thread_local cudaEvent_t cuda_memory_bandwidth_profile_end_event = nullptr;

}  // namespace

void TraceKernelForwardDataContentStart(
    const Kernel*, const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
#if defined(WITH_CUDA)
  if (profile_cuda_memory_bandwidth) {
    CHECK(cuda_memory_bandwidth_profile_start_event == nullptr);
    CHECK(cuda_memory_bandwidth_profile_end_event == nullptr);
    auto* cuda_device_ctx = dynamic_cast<CudaDeviceCtx*>(ctx.device_ctx);
    if (cuda_device_ctx) {
      OF_CUDA_CHECK(cudaEventCreate(&cuda_memory_bandwidth_profile_start_event));
      OF_CUDA_CHECK(cudaEventCreate(&cuda_memory_bandwidth_profile_end_event));
      OF_CUDA_CHECK(cudaEventRecord(cuda_memory_bandwidth_profile_start_event,
                                    cuda_device_ctx->cuda_stream()));
    }
  }
#endif  // WITH_CUDA
}

void TraceKernelForwardDataContentEnd(const Kernel* kernel, const KernelCtx& ctx,
                                      const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
#if defined(WITH_CUDA)
  if (profile_cuda_memory_bandwidth) {
    auto* cuda_device_ctx = dynamic_cast<CudaDeviceCtx*>(ctx.device_ctx);
    if (cuda_device_ctx) {
      CHECK_NOTNULL(cuda_memory_bandwidth_profile_start_event);
      CHECK_NOTNULL(cuda_memory_bandwidth_profile_end_event);
      OF_CUDA_CHECK(
          cudaEventRecord(cuda_memory_bandwidth_profile_end_event, cuda_device_ctx->cuda_stream()));
      OF_CUDA_CHECK(cudaStreamSynchronize(cuda_device_ctx->cuda_stream()));
      float elapsed_ms;
      OF_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, cuda_memory_bandwidth_profile_start_event,
                                         cuda_memory_bandwidth_profile_end_event));
      OF_CUDA_CHECK(cudaEventDestroy(cuda_memory_bandwidth_profile_start_event));
      OF_CUDA_CHECK(cudaEventDestroy(cuda_memory_bandwidth_profile_end_event));
      cuda_memory_bandwidth_profile_start_event = nullptr;
      cuda_memory_bandwidth_profile_end_event = nullptr;
      int64_t memory_size = 0;
      for (const auto& bn : kernel->op_attribute().input_bns()) {
        const Blob* blob = BnInOp2Blob(bn);
        if (blob) { memory_size += blob->ByteSizeOfBlobBody(); }
      }
      for (const auto& bn : kernel->op_attribute().output_bns()) {
        const Blob* blob = BnInOp2Blob(bn);
        if (blob) { memory_size += blob->ByteSizeOfBlobBody(); }
      }
      double bandwidth =
          static_cast<double>(memory_size) / (1024.0 * 1024.0 * 1024.0) / (elapsed_ms / 1000);
      LOG(INFO) << "PROFILER::KERNEL::CUDA_MEMORY_BANDWIDTH op_name: " << kernel->op_conf().name()
                << " elapsed(ms): " << elapsed_ms << " memory_size(Byte): " << memory_size
                << " bandwidth(GB/s): " << bandwidth;
    }
  }
#endif  // WITH_CUDA
}

}  // namespace profiler

}  // namespace oneflow