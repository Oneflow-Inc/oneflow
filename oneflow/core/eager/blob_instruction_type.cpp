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
#include "oneflow/core/object_msg/flat_msg_view.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/eager/blob_instruction_type.h"
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/vm/device_helper_stream_type.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/eager/lazy_ref_blob_object.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/vm/operand_def.h"

namespace oneflow {
namespace eager {

namespace {

// clang-format off
FLAT_MSG_VIEW_BEGIN(PinBlobInstruction);
  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::MutOperand, blob);
FLAT_MSG_VIEW_END(PinBlobInstruction);
// clang-format on

}  // namespace

#ifdef WITH_CUDA
class CudaHostRegisterBlobInstructionType final : public vm::InstructionType {
 public:
  CudaHostRegisterBlobInstructionType() = default;
  ~CudaHostRegisterBlobInstructionType() override = default;

  using stream_type = vm::DeviceHelperStreamType;

  void Infer(vm::Instruction* instruction) const override {
    // do nothing
  }
  void Compute(vm::Instruction* instruction) const override {
    FlatMsgView<PinBlobInstruction> args(instruction->instr_msg().operand());
    auto* blob_obj = CHECK_JUST(instruction->mut_operand_type(args->blob())->Mut<BlobObject>());
    auto* blob = blob_obj->mut_blob();
    CHECK(blob->mem_case().has_host_mem());
    if (blob->mem_case().host_mem().has_cuda_pinned_mem()) { return; }
    void* dptr = blob->mut_dptr();
    CHECK_NOTNULL(dptr);
    size_t size = blob->AlignedByteSizeOfBlobBody();
    cudaError_t cuda_error = cudaHostRegister(dptr, size, cudaHostRegisterDefault);
    if (cuda_error == cudaErrorHostMemoryAlreadyRegistered) { return; }
    OF_CUDA_CHECK(cuda_error);
  }
};
COMMAND(vm::RegisterInstructionType<CudaHostRegisterBlobInstructionType>("CudaHostRegisterBlob"));

class CudaHostUnregisterBlobInstructionType final : public vm::InstructionType {
 public:
  CudaHostUnregisterBlobInstructionType() = default;
  ~CudaHostUnregisterBlobInstructionType() override = default;

  using stream_type = vm::DeviceHelperStreamType;

  void Infer(vm::Instruction* instruction) const override {
    // do nothing
  }
  void Compute(vm::Instruction* instruction) const override {
    FlatMsgView<PinBlobInstruction> args(instruction->instr_msg().operand());
    auto* blob_obj = CHECK_JUST(instruction->mut_operand_type(args->blob())->Mut<BlobObject>());
    auto* blob = blob_obj->mut_blob();
    CHECK(blob->mem_case().has_host_mem());
    if (blob->mem_case().host_mem().has_cuda_pinned_mem()) { return; }
    void* dptr = blob->mut_dptr();
    CHECK_NOTNULL(dptr);
    cudaError_t cuda_error = cudaHostUnregister(dptr);
    if (cuda_error == cudaErrorHostMemoryNotRegistered) { return; }
    OF_CUDA_CHECK(cuda_error);
  }
};
COMMAND(
    vm::RegisterInstructionType<CudaHostUnregisterBlobInstructionType>("CudaHostUnregisterBlob"));
#endif

Maybe<void> LazyReferenceInstructionType::Run(vm::Instruction* instruction) const {
  FlatMsgView<LazyReferenceInstruction> args(instruction->instr_msg().operand());
  vm::RwMutexedObject* eager_blob_rw = instruction->mut_operand_type(args->eager_blob());
  const auto* lbn_operand = instruction->operand_type(args->lbn_sym_id());
  const auto lbn = JUST(lbn_operand->template Get<vm::StringObject>()).str();
  ParallelContext parallel_ctx;
  JUST(instruction->parallel_desc()->GetParallelContext(
      &parallel_ctx, instruction->stream().machine_id(), instruction->stream().device_id()));
  Blob* blob = Global<RegstMgr>::Get()->Blob4LbiAndParallelId(GenLogicalBlobId(lbn),
                                                              parallel_ctx.parallel_id());
  eager_blob_rw->Init<eager::LazyRefBlobObject>(blob);
  return Maybe<void>::Ok();
}

}  // namespace eager
}  // namespace oneflow
