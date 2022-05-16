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
#ifdef WITH_CUDA

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/eager/opkernel_instruction_type.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/async_cuda_stream_type.h"
#include "oneflow/core/vm/cuda_copy_h2d_stream_type.h"
#include "oneflow/core/vm/cuda_copy_d2h_stream_type.h"
#include "oneflow/core/vm/instruction.h"

namespace oneflow {
namespace vm {

class CudaLocalCallOpKernelInstructionType final : public LocalCallOpKernelInstructionType {
 public:
  CudaLocalCallOpKernelInstructionType() = default;
  ~CudaLocalCallOpKernelInstructionType() override = default;

  using stream_type = vm::CudaStreamType;
};
COMMAND(
    vm::RegisterInstructionType<CudaLocalCallOpKernelInstructionType>("cuda.LocalCallOpKernel"));

class AsyncCudaLocalCallOpKernelInstructionType final : public LocalCallOpKernelInstructionType {
 public:
  AsyncCudaLocalCallOpKernelInstructionType() = default;
  ~AsyncCudaLocalCallOpKernelInstructionType() override = default;

  using stream_type = vm::AsyncCudaStreamType;
};
COMMAND(vm::RegisterInstructionType<AsyncCudaLocalCallOpKernelInstructionType>(
    "async.cuda.LocalCallOpKernel"));

class CudaH2DLocalCallOpKernelInstructionType final : public LocalCallOpKernelInstructionType {
 public:
  CudaH2DLocalCallOpKernelInstructionType() = default;
  ~CudaH2DLocalCallOpKernelInstructionType() override = default;

  using stream_type = vm::CudaCopyH2DStreamType;
};
COMMAND(vm::RegisterInstructionType<CudaH2DLocalCallOpKernelInstructionType>(
    "cuda_h2d.LocalCallOpKernel"));

class CudaD2HLocalCallOpKernelInstructionType final : public LocalCallOpKernelInstructionType {
 public:
  CudaD2HLocalCallOpKernelInstructionType() = default;
  ~CudaD2HLocalCallOpKernelInstructionType() override = default;

  using stream_type = vm::CudaCopyD2HStreamType;
};
COMMAND(vm::RegisterInstructionType<CudaD2HLocalCallOpKernelInstructionType>(
    "cuda_d2h.LocalCallOpKernel"));

}  // namespace vm
}  // namespace oneflow

#endif
