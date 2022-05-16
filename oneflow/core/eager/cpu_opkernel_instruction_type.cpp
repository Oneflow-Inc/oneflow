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
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/eager/opkernel_instruction_type.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/cpu_stream_type.h"
#include "oneflow/core/vm/instruction.h"

namespace oneflow {
namespace vm {

class CpuLocalCallOpKernelInstructionType final : public LocalCallOpKernelInstructionType {
 public:
  CpuLocalCallOpKernelInstructionType() = default;
  ~CpuLocalCallOpKernelInstructionType() override = default;

  using stream_type = vm::CpuStreamType;
};
COMMAND(vm::RegisterInstructionType<CpuLocalCallOpKernelInstructionType>("cpu.LocalCallOpKernel"));

}  // namespace vm
}  // namespace oneflow
