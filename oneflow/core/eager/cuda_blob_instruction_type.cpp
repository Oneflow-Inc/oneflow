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
#include "oneflow/core/eager/blob_instruction_type.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/async_cuda_stream_type.h"

namespace oneflow {
namespace vm {
class GpuLazyReferenceInstructionType : public LazyReferenceInstructionType {
 public:
  GpuLazyReferenceInstructionType() = default;
  ~GpuLazyReferenceInstructionType() override = default;

  using stream_type = vm::AsyncCudaStreamType;
};
COMMAND(vm::RegisterInstructionType<GpuLazyReferenceInstructionType>("gpu.LazyReference"));

class GpuAccessBlobByCallbackInstructionType final : public AccessBlobByCallbackInstructionType {
 public:
  GpuAccessBlobByCallbackInstructionType() = default;
  ~GpuAccessBlobByCallbackInstructionType() override = default;
  using stream_type = vm::CudaStreamType;
};
COMMAND(vm::RegisterInstructionType<GpuAccessBlobByCallbackInstructionType>(
    "gpu.AccessBlobByCallback"));

class GpuSoftSyncStreamInstructionType : public SoftSyncStreamInstructionType {
 public:
  GpuSoftSyncStreamInstructionType() = default;
  ~GpuSoftSyncStreamInstructionType() override = default;
  using stream_type = vm::CudaStreamType;
};
COMMAND(vm::RegisterInstructionType<GpuSoftSyncStreamInstructionType>("gpu.SoftSyncStream"));

}  // namespace vm
}  // namespace oneflow
#endif
