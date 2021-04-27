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
#include "oneflow/core/eager/opkernel_object.h"
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/eager/opkernel_instruction.msg.h"
#include "oneflow/core/eager/opkernel_instruction_type.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/cpu_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/object.h"

namespace oneflow {
namespace eager {

class CpuLocalCallOpKernelInstructionType final : public LocalCallOpKernelInstructionType {
 public:
  CpuLocalCallOpKernelInstructionType() = default;
  ~CpuLocalCallOpKernelInstructionType() override = default;

  using stream_type = vm::CpuStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<CpuLocalCallOpKernelInstructionType>("cpu.LocalCallOpKernel"));

class CpuCallOpKernelInstructionType final : public CallOpKernelInstructionType {
 public:
  CpuCallOpKernelInstructionType() = default;
  ~CpuCallOpKernelInstructionType() override = default;

  using stream_type = vm::CpuStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<CpuCallOpKernelInstructionType>("cpu.CallOpKernel"));

class CpuUserStatelessCallOpKernelInstructionType final
    : public UserStatelessCallOpKernelInstructionType {
 public:
  CpuUserStatelessCallOpKernelInstructionType() = default;
  ~CpuUserStatelessCallOpKernelInstructionType() override = default;

  using stream_type = vm::CpuStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<CpuUserStatelessCallOpKernelInstructionType>(
    "cpu.compute.UserStatelessCallOpKernel"));

class CpuSystemStatelessCallOpKernelInstructionType final
    : public SystemStatelessCallOpKernelInstructionType {
 public:
  CpuSystemStatelessCallOpKernelInstructionType() = default;
  ~CpuSystemStatelessCallOpKernelInstructionType() override = default;

  using stream_type = vm::CpuStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<CpuSystemStatelessCallOpKernelInstructionType>(
    "cpu.compute.SystemStatelessCallOpKernel"));

class CpuFetchBlobHeaderInstructionType final : public FetchBlobHeaderInstructionType {
 public:
  CpuFetchBlobHeaderInstructionType() = default;
  ~CpuFetchBlobHeaderInstructionType() override = default;

  using stream_type = vm::CpuStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<CpuFetchBlobHeaderInstructionType>("cpu.FetchBlobHeader"));

class CpuFetchBlobBodyInstructionType final : public FetchBlobBodyInstructionType {
 public:
  CpuFetchBlobBodyInstructionType() = default;
  ~CpuFetchBlobBodyInstructionType() override = default;

  using stream_type = vm::CpuStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<CpuFetchBlobBodyInstructionType>("cpu.FetchBlobBody"));

class CpuFeedBlobInstructionType final : public FeedBlobInstructionType {
 public:
  CpuFeedBlobInstructionType() = default;
  ~CpuFeedBlobInstructionType() override = default;

  using stream_type = vm::CpuStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<CpuFeedBlobInstructionType>("cpu.FeedBlob"));

}  // namespace eager
}  // namespace oneflow
