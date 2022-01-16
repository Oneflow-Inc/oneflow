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
#include "oneflow/core/eager/release_tensor_instruction_type.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/release_tensor_arg_phy_instr_operand.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/cpu_stream_type.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/dtr_cuda_allocator.h"
#include "oneflow/core/vm/thread_safe_allocator.h"

namespace oneflow {

namespace vm {

void EvictDTRTensorInstructionType::Infer(vm::Instruction* instruction) const { UNIMPLEMENTED(); }

void EvictDTRTensorInstructionType::Compute(vm::Instruction* instruction) const {
  const vm::InstructionMsg& instr_msg = instruction->instr_msg();
  const auto& phy_instr_operand = instr_msg.phy_instr_operand();
  CHECK(static_cast<bool>(phy_instr_operand));
  const auto* ptr =
      dynamic_cast<const vm::ReleaseTensorArgPhyInstrOperand*>(phy_instr_operand.get());
  CHECK_NOTNULL(ptr);
  if (std::getenv("OF_DTR_NO_EE") == nullptr) {
    if (oneflow::DTRDebugEnabled()) {
      std::cout << "eager eviction tensor " << ptr->eager_blob_object().get() << " with ref count "
                << ptr->eager_blob_object().use_count() << std::endl;
    }
    const auto& ebo =
        CHECK_NOTNULL(std::dynamic_pointer_cast<vm::DTREagerBlobObject>(ptr->eager_blob_object()));
    if (!ebo->is_evictable()) {
      // std::cout << "skip because non evictable" << std::endl;
    } else if (ebo->is_pinned()) {
      // std::cout << "skip because pinned" << std::endl;
    } else {
      CHECK_JUST(ebo->evict());
    }
  }
}

class CpuEvictDTRTensorInstructionType final : public EvictDTRTensorInstructionType {
 public:
  CpuEvictDTRTensorInstructionType() = default;
  ~CpuEvictDTRTensorInstructionType() override = default;
  using stream_type = vm::CpuStreamType;
};
COMMAND(vm::RegisterInstructionType<CpuEvictDTRTensorInstructionType>("cpu.EvictDTRTensor"));

#ifdef WITH_CUDA
class GpuEvictDTRTensorInstructionType final : public EvictDTRTensorInstructionType {
 public:
  GpuEvictDTRTensorInstructionType() = default;
  ~GpuEvictDTRTensorInstructionType() override = default;
  using stream_type = vm::CudaStreamType;
};
COMMAND(vm::RegisterInstructionType<GpuEvictDTRTensorInstructionType>("gpu.EvictDTRTensor"));
#endif

void ReleaseTensorInstructionType::Infer(vm::Instruction* instruction) const { UNIMPLEMENTED(); }

void ReleaseTensorInstructionType::Compute(vm::Instruction* instruction) const {
  const vm::InstructionMsg& instr_msg = instruction->instr_msg();
  const auto& phy_instr_operand = instr_msg.phy_instr_operand();
  CHECK(static_cast<bool>(phy_instr_operand));
  const auto* ptr =
      dynamic_cast<const vm::ReleaseTensorArgPhyInstrOperand*>(phy_instr_operand.get());
  CHECK_NOTNULL(ptr);
  if (oneflow::DTRDebugEnabled()) {
    LOG(INFO) << "release tensor " << ptr->eager_blob_object().get() << " with ref count "
              << ptr->eager_blob_object().use_count() << std::endl;
  }
  CHECK_JUST(ptr->eager_blob_object()->DeallocateBlobDataPtr());
}

class CpuReleaseTensorInstructionType final : public ReleaseTensorInstructionType {
 public:
  CpuReleaseTensorInstructionType() = default;
  ~CpuReleaseTensorInstructionType() override = default;
  using stream_type = vm::CpuStreamType;
};
COMMAND(vm::RegisterInstructionType<CpuReleaseTensorInstructionType>("cpu.ReleaseTensor"));

#ifdef WITH_CUDA
class GpuReleaseTensorInstructionType final : public ReleaseTensorInstructionType {
 public:
  GpuReleaseTensorInstructionType() = default;
  ~GpuReleaseTensorInstructionType() override = default;
  using stream_type = vm::CudaStreamType;
};
COMMAND(vm::RegisterInstructionType<GpuReleaseTensorInstructionType>("gpu.ReleaseTensor"));
#endif

void TempInstructionType::Infer(vm::Instruction* instruction) const { UNIMPLEMENTED(); }

template<typename T, typename U>
T dynamic_cast_with_check(U* ptr) {
  CHECK_NOTNULL(ptr);
  T ret = dynamic_cast<T>(ptr);
  if (ret == nullptr) {
    LOG(FATAL) << "dynamic_cast failed, real type " << typeid(*ptr).name() << ", target type "
               << typeid(T).name();
  }
  return ret;
}

void TempInstructionType::Compute(vm::Instruction* instruction) const {
  auto* allocator = dynamic_cast_with_check<vm::DtrCudaAllocator*>(
      dynamic_cast_with_check<vm::ThreadSafeAllocator*>(
          instruction->stream().device_ctx()->mut_allocator())
          ->backend_allocator());
  allocator->DisplayAllPieces();
}
COMMAND(vm::RegisterInstructionType<TempInstructionType>("Temp"));

}  // namespace vm
}  // namespace oneflow
