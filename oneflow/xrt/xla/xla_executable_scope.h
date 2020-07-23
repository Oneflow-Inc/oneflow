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
#include "tensorflow/compiler/jit/xla_lib/xla_runtime_util.h"

namespace oneflow {
namespace xrt {
namespace mola {

inline bool SupportMultiStream(const XrtDevice &device) {
  switch (device) {
    case XrtDevice::CPU_X86: return false;
    case XrtDevice::GPU_CUDA: return true;
    default: {
      LOG(FATAL) << "Unknow device " << device;
      return false;
    }
  }
}

class XlaExecutableRunScope {
 public:
  inline XlaExecutableRunScope(xla::LocalExecutable *executable,
                               XlaExecutableRunContext &run_context);

  inline virtual ~XlaExecutableRunScope();

 private:
  void *launch_stream_ = nullptr;
  XlaExecutableRunContext &run_context_;
};

XlaExecutableRunScope::XlaExecutableRunScope(xla::LocalExecutable *executable,
                                             XlaExecutableRunContext &run_context)
    : run_context_(run_context) {
  // Swap cuda stream between the backend stream and context, so XLA could
  // launch kernel on the specified cuda stream of the context. Note that it
  // should do nothing for single stream device such as CPU.
  launch_stream_ = run_context_.run_options().stream;
  if (SupportMultiStream(run_context_.device())) {
    xla::SwapGpuStreamHandle(run_context_.stream(), &launch_stream_);
  }

  size_t workspace_size = xla::CalcWorkspaceByteSize(executable);
  run_context_.ReserveWorkspace(workspace_size);
  run_context_.LockWorkspace();
}

XlaExecutableRunScope::~XlaExecutableRunScope() {
  if (SupportMultiStream(run_context_.device())) {
    xla::SwapGpuStreamHandle(run_context_.stream(), &launch_stream_);
  }
  run_context_.UnlockWorkspace();
}

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_XLA_EXECUTABLE_SCOPE_H_
