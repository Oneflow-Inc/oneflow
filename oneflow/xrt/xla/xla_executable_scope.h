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
#ifndef ONEFLOW_XRT_XLA_XLA_EXECUTABLE_SCOPE_H_
#define ONEFLOW_XRT_XLA_XLA_EXECUTABLE_SCOPE_H_

#include "oneflow/xrt/xla/xla_executable_context.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "tensorflow/compiler/jit/xla_lib/xla_runtime_util.h"

namespace oneflow {
namespace xrt {
namespace mola {

static void SwapStream(const XrtDevice& device, ep::Stream* stream, se::Stream* xla_stream) {
  switch (device) {
    case XrtDevice::GPU_CUDA: {
#ifdef WITH_CUDA
      auto* cuda_stream = stream->As<ep::CudaStream>()->cuda_stream();
      xla::SwapGpuStreamHandle(xla_stream, (void**)&cuda_stream);
#else
      LOG(FATAL) << "CUDA is not supported, please recompile oneflow with WITH_CUDA=ON" << device;
#endif  // WITH_CUDA
      break;
    }
    default: {
      break;
    }
  }
}

class XlaExecutableRunScope {
 public:
  inline XlaExecutableRunScope(xla::LocalExecutable* executable,
                               XlaExecutableRunContext& run_context);

  inline virtual ~XlaExecutableRunScope();

 private:
  ep::Stream* stream_ = nullptr;
  XlaExecutableRunContext& run_context_;
};

XlaExecutableRunScope::XlaExecutableRunScope(xla::LocalExecutable* executable,
                                             XlaExecutableRunContext& run_context)
    : run_context_(run_context) {
  // Swap cuda stream between the backend stream and context, so XLA could
  // launch kernel on the specified cuda stream of the context. Note that it
  // should do nothing for single stream device such as CPU.
  stream_ = run_context_.run_options().stream;
  if (stream_) { SwapStream(run_context_.device(), stream_, run_context_.stream()); }
  size_t workspace_size = xla::CalcWorkspaceByteSize(executable);
  run_context_.ReserveWorkspace(workspace_size);
  run_context_.LockWorkspace();
}

XlaExecutableRunScope::~XlaExecutableRunScope() {
  if (stream_) { SwapStream(run_context_.device(), stream_, run_context_.stream()); }
  run_context_.UnlockWorkspace();
}

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_XLA_EXECUTABLE_SCOPE_H_
