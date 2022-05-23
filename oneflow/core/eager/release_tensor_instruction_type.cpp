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
#include "oneflow/core/eager/dtr_util.h"
#include "oneflow/core/eager/dtr_eager_blob_object.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/async_cuda_stream_type.h"
#include "oneflow/core/vm/cuda_copy_h2d_stream_type.h"
#include "oneflow/core/vm/cuda_copy_d2h_stream_type.h"
#include "oneflow/core/vm/cpu_stream_type.h"
#include "oneflow/core/vm/cuda_optional_event_record_status_querier.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/dtr_cuda_allocator.h"
#include "oneflow/core/vm/thread_safe_allocator.h"

namespace oneflow {

namespace vm {

class EvictDTRTensorInstructionType : public vm::InstructionType {
 public:
  EvictDTRTensorInstructionType() = default;
  ~EvictDTRTensorInstructionType() override = default;

  void Compute(vm::Instruction* instruction) const override;
};

void EvictDTRTensorInstructionType::Compute(vm::Instruction* instruction) const {
  const vm::InstructionMsg& instr_msg = instruction->instr_msg();
  const auto& phy_instr_operand = instr_msg.phy_instr_operand();
  CHECK(static_cast<bool>(phy_instr_operand));
  const auto* ptr =
      dynamic_cast<const vm::ReleaseTensorArgPhyInstrOperand*>(phy_instr_operand.get());
  CHECK_NOTNULL(ptr);
  if (ParseBooleanFromEnv("OF_DTR_EE", true)) {
    const auto& ebo =
        CHECK_NOTNULL(std::dynamic_pointer_cast<vm::DTREagerBlobObject>(ptr->eager_blob_object()));
    if (dtr::debug_level() >= 2) {
      LOG(INFO) << "eager eviction tensor " << ebo.get() << " of op " << ebo->compute_op_type_name()
                << " with size " << ebo->BlobBodyBytes();
    }
    if (!ebo->is_evictable()) {
      if (dtr::debug_level() >= 2) { LOG(INFO) << "but skip because non evictable" << std::endl; }
    } else if (ebo->is_pinned()) {
      if (dtr::debug_level() >= 2) { LOG(INFO) << "but skip because pinned" << std::endl; }
    } else {
      CHECK_JUST(ebo->evict(true));
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
COMMAND(vm::RegisterInstructionType<GpuEvictDTRTensorInstructionType>("cuda.EvictDTRTensor"));
#endif

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
    if (dtr::is_enabled_and_debug()) {
      LOG(INFO) << "ReleaseTensor instruction: " << ptr->eager_blob_object().get() << "(id: "
                << std::dynamic_pointer_cast<vm::DTREagerBlobObject>(ptr->eager_blob_object())->id()
                << ") with ref count " << ptr->eager_blob_object().use_count() << std::endl;
    }
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

class TempInstructionType : public vm::InstructionType {
 public:
  TempInstructionType() = default;
  ~TempInstructionType() override = default;

  using stream_type = vm::CudaStreamType;

  void Compute(vm::Instruction* instruction) const override;
};

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
