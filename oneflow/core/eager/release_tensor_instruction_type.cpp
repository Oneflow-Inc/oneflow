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
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/eager/release_tensor_arg_phy_instr_operand.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/async_cuda_stream_type.h"
#include "oneflow/core/vm/cuda_copy_h2d_stream_type.h"
#include "oneflow/core/vm/cuda_copy_d2h_stream_type.h"
#include "oneflow/core/vm/cpu_stream_type.h"
#include "oneflow/core/vm/cuda_optional_event_record_status_querier.h"

namespace oneflow {

namespace vm {

template<typename StreamT>
class ReleaseTensorInstructionType : public vm::InstructionType {
 public:
  ReleaseTensorInstructionType() = default;
  ~ReleaseTensorInstructionType() override = default;

  using stream_type = StreamT;

  InstructionFuseType fuse_type() const override { return kEnableInstructionFuseAtAnyPosition; }

  void Release(const vm::InstructionMsg& instr_msg) const {
    const auto& phy_instr_operand = instr_msg.phy_instr_operand();
    CHECK(static_cast<bool>(phy_instr_operand));
    const auto* ptr =
        dynamic_cast<const vm::ReleaseTensorArgPhyInstrOperand*>(phy_instr_operand.get());
    CHECK_NOTNULL(ptr);
    CHECK_JUST(ptr->eager_blob_object()->DeallocateBlobDataPtr());
  }
  void Compute(vm::Instruction* instruction) const override { Release(instruction->instr_msg()); }
  void ComputeInFuseMode(vm::InstructionMsg* instr_msg) const override { Release(*instr_msg); }
};

COMMAND(
    vm::RegisterInstructionType<ReleaseTensorInstructionType<CpuStreamType>>("cpu.ReleaseTensor"));
COMMAND(vm::RegisterInstructionType<ReleaseTensorInstructionType<CpuStreamType>>(
    "comm_net.ReleaseTensor"));

#ifdef WITH_CUDA

template<typename StreamT>
class CudaReleaseTensorInstructionType : public ReleaseTensorInstructionType<StreamT> {
 public:
  CudaReleaseTensorInstructionType() = default;
  ~CudaReleaseTensorInstructionType() override = default;

  void InitInstructionStatus(Instruction* instruction) const override {
    auto* status_buffer = instruction->mut_status_buffer();
    auto* stream = instruction->mut_stream();
    instruction->stream_type().InitInstructionStatus(*stream, status_buffer);
    auto* data_ptr = status_buffer->mut_buffer()->mut_data();
    CudaOptionalEventRecordStatusQuerier::MutCast(data_ptr)->reset_cuda_event(nullptr);
  }
};

COMMAND(vm::RegisterInstructionType<CudaReleaseTensorInstructionType<CudaStreamType>>(
    "gpu.ReleaseTensor"));
COMMAND(vm::RegisterInstructionType<CudaReleaseTensorInstructionType<CudaStreamType>>(
    "cuda.ReleaseTensor"));
COMMAND(vm::RegisterInstructionType<CudaReleaseTensorInstructionType<CudaCopyH2DStreamType>>(
    "cuda_h2d.ReleaseTensor"));
COMMAND(vm::RegisterInstructionType<CudaReleaseTensorInstructionType<CudaCopyD2HStreamType>>(
    "cuda_d2h.ReleaseTensor"));
COMMAND(vm::RegisterInstructionType<CudaReleaseTensorInstructionType<CudaStreamType>>(
    "sync_launched_nccl.ReleaseTensor"));
COMMAND(vm::RegisterInstructionType<CudaReleaseTensorInstructionType<AsyncCudaStreamType>>(
    "async_launched_nccl.ReleaseTensor"));
#endif

}  // namespace vm
}  // namespace oneflow
