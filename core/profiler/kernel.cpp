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
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/lazy/actor/actor_context.h"

namespace oneflow {

namespace profiler {

namespace {

bool profile_cuda_memory_bandwidth = false;
bool profile_kernel_forward_range = false;

void Init() {
  profile_cuda_memory_bandwidth =
      ParseBooleanFromEnv("ONEFLOW_PROFILER_KERNEL_PROFILE_CUDA_MEMORY_BANDWIDTH", false);
  profile_kernel_forward_range =
      ParseBooleanFromEnv("ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE", false);
}

COMMAND(Init());

#if defined(WITH_CUDA)
thread_local cudaEvent_t cuda_memory_bandwidth_profile_start_event = nullptr;
thread_local cudaEvent_t cuda_memory_bandwidth_profile_end_event = nullptr;
#endif  // WITH_CUDA

}  // namespace

void TraceKernelForwardDataContentStart(KernelContext* kernel_ctx, const Kernel* kernel) {
#if defined(WITH_CUDA)
  if (profile_cuda_memory_bandwidth) {
    auto* actor_context_provider = dynamic_cast<ActorContextProvider*>(kernel_ctx);
    auto* cuda_stream = dynamic_cast<ep::CudaStream*>(kernel_ctx->stream());
    if (cuda_stream != nullptr && actor_context_provider != nullptr) {
      CHECK(cuda_memory_bandwidth_profile_start_event == nullptr);
      CHECK(cuda_memory_bandwidth_profile_end_event == nullptr);
      OF_CUDA_CHECK(cudaEventCreate(&cuda_memory_bandwidth_profile_start_event));
      OF_CUDA_CHECK(cudaEventCreate(&cuda_memory_bandwidth_profile_end_event));
      OF_CUDA_CHECK(
          cudaEventRecord(cuda_memory_bandwidth_profile_start_event, cuda_stream->cuda_stream()));
    }
  }
  if (profile_kernel_forward_range) { OF_PROFILER_RANGE_PUSH(kernel->op_conf().name()); }
#endif  // WITH_CUDA
}

void TraceKernelForwardDataContentEnd(KernelContext* kernel_ctx, const Kernel* kernel) {
#if defined(WITH_CUDA)
  if (profile_kernel_forward_range) { OF_PROFILER_RANGE_POP(); }
  // The memory bandwidth profiler only works in lazy mode.
  if (profile_cuda_memory_bandwidth) {
    auto* cuda_stream = dynamic_cast<ep::CudaStream*>(kernel_ctx->stream());
    auto* actor_context_provider = dynamic_cast<ActorContextProvider*>(kernel_ctx);
    if (cuda_stream != nullptr && actor_context_provider != nullptr) {
      cudaEvent_t start_event = cuda_memory_bandwidth_profile_start_event;
      cudaEvent_t end_event = cuda_memory_bandwidth_profile_end_event;
      cuda_memory_bandwidth_profile_start_event = nullptr;
      cuda_memory_bandwidth_profile_end_event = nullptr;
      CHECK_NOTNULL(start_event);
      CHECK_NOTNULL(end_event);
      OF_CUDA_CHECK(cudaEventRecord(end_event, cuda_stream->cuda_stream()));
      int64_t memory_size = 0;
      for (const auto& bn : kernel->op_attribute().input_bns()) {
        const Blob* blob = kernel_ctx->BnInOp2Blob(bn);
        if (blob) { memory_size += blob->ByteSizeOfBlobBody(); }
      }
      for (const auto& bn : kernel->op_attribute().output_bns()) {
        const Blob* blob = kernel_ctx->BnInOp2Blob(bn);
        if (blob) { memory_size += blob->ByteSizeOfBlobBody(); }
      }
      const std::string op_name = kernel->op_conf().name();
      actor_context_provider->GetActorContext()->AddCallback(
          [start_event, end_event, memory_size, op_name]() {
            float elapsed_ms = 0;
            OF_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, end_event));
            OF_CUDA_CHECK(cudaEventDestroy(start_event));
            OF_CUDA_CHECK(cudaEventDestroy(end_event));
            double bandwidth =
                static_cast<double>(memory_size) / (1024.0 * 1024.0 * 1024.0) / (elapsed_ms / 1000);
            LOG(INFO) << "PROFILER::KERNEL::CUDA_MEMORY_BANDWIDTH op_name: " << op_name
                      << " elapsed(ms): " << elapsed_ms << " memory_size(Byte): " << memory_size
                      << " bandwidth(GB/s): " << bandwidth;
          });
    }
  }
#endif  // WITH_CUDA
}

}  // namespace profiler

}  // namespace oneflow
