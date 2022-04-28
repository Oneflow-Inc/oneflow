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
#ifdef WITH_NPU

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/eager/opkernel_instruction_type.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/npu_copy_h2d_stream_type.h"
#include "oneflow/core/vm/npu_copy_d2h_stream_type.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/npu_stream_type.h"
namespace oneflow {
namespace vm {

class NpuH2DLocalCallOpKernelInstructionType final : public LocalCallOpKernelInstructionType {
 public:
  NpuH2DLocalCallOpKernelInstructionType() = default;
  ~NpuH2DLocalCallOpKernelInstructionType() override = default;

  using stream_type = vm::NpuCopyH2DStreamType;
};
COMMAND(vm::RegisterInstructionType<NpuH2DLocalCallOpKernelInstructionType>(
    "npu_h2d.LocalCallOpKernel"));

class NpuD2HLocalCallOpKernelInstructionType final : public LocalCallOpKernelInstructionType {
 public:
  NpuD2HLocalCallOpKernelInstructionType() = default;
  ~NpuD2HLocalCallOpKernelInstructionType() override = default;

  using stream_type = vm::NpuCopyD2HStreamType;
};
COMMAND(vm::RegisterInstructionType<NpuD2HLocalCallOpKernelInstructionType>(
    "npu_d2h.LocalCallOpKernel"));

class NpuLocalCallOpKernelInstructionType final : public LocalCallOpKernelInstructionType {
 public:
  NpuLocalCallOpKernelInstructionType() = default;
  ~NpuLocalCallOpKernelInstructionType() override = default;

  using stream_type = vm::NpuStreamType;
};
COMMAND(vm::RegisterInstructionType<NpuLocalCallOpKernelInstructionType>("npu.LocalCallOpKernel"));
}  // namespace vm
}  // namespace oneflow

#endif
