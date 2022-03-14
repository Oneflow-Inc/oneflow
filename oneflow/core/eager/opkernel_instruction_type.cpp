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
#include <algorithm>
#include <iterator>
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/eager/dtr_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/dtr_cuda_allocator.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/eager/opkernel_instruction.h"
#include "oneflow/core/eager/opkernel_instruction_type.h"
#include "oneflow/core/eager/local_call_opkernel_phy_instr_operand.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/object.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/vm/thread_safe_allocator.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/framework/tensor_pool.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"

namespace oneflow {
namespace vm {

using DTREagerBlobObjectList = std::vector<std::shared_ptr<vm::DTREagerBlobObject>>;

Maybe<LocalCallOpKernelPhyInstrOperand> GetLocalCallOpKernelPhyInstrOperand(
    vm::Instruction* instruction) {
  const auto& operand = instruction->instr_msg().phy_instr_operand();
  CHECK_OR_RETURN(static_cast<bool>(operand));
  auto ptr = std::dynamic_pointer_cast<LocalCallOpKernelPhyInstrOperand>(operand);
  CHECK_NOTNULL_OR_RETURN(ptr);
  return ptr;
}

struct LocalCallOpKernelUtil {
  // inputs are guaranteed to be in memory before calling this function,
  // thus no recomputation is needed, but eviction is still possible.
  static inline Maybe<void> ComputeOperand(LocalCallOpKernelPhyInstrOperand* operand,
                                           DeviceCtx* device_ctx) {
    operand->mut_opkernel()->composed_attrs_for_scheduler_thread()->ResetPrior(operand->attrs());
    operand->set_user_opkernel(JUST(operand->mut_opkernel()->ChooseOpKernel(
        operand->inputs(), operand->outputs(), operand->consistent_tensor_infer_result())));
    JUST(InitOutputBlobs(operand));

    JUST(InferTempStorageBlobDesc(operand));
    JUST(ResetTempStorageBlob(operand));

    JUST(AllocateOutputBlobsMemory(operand, device_ctx));
    JUST(TryAllocateTempStorageBlobMemory(operand, device_ctx));

    user_op::OpKernelState* state;
    TryInitOpKernelState(operand, device_ctx, &state);
    if (DTREnabled()) { JUST(CheckInputInMemory(operand)); }
    JUST(OpKernelCompute(operand, device_ctx, state));

    JUST(DeallocateTempStorageBlobMemory(operand, device_ctx));
    operand->set_user_opkernel(nullptr);
    if (DTRDebugEnabled()) {
      LOG(INFO) << operand->shared_opkernel()->op_type_name() << " ComputeOperand done";
    }
    if (DTREnabled()) { JUST(CheckOutputInMemory(operand)); }
    return Maybe<void>::Ok();
  }
  static inline Maybe<void> ComputeInstruction(vm::Instruction* instruction);

 private:
  static inline Maybe<const MemoryCase&> GetMemCase(LocalCallOpKernelPhyInstrOperand* operand) {
    const auto& mem_case = operand->opkernel().mem_case();
    CHECK_OR_RETURN(static_cast<bool>(mem_case));
    return *mem_case;
  }

  static inline Maybe<void> CheckMemCase(const MemoryCase& mem_case, DeviceType device_type,
                                         int64_t device_id) {
    if (mem_case.has_host_mem()) {
      CHECK_EQ_OR_RETURN(device_type, DeviceType::kCPU);
    } else if (mem_case.has_device_cuda_mem()) {
      CHECK_EQ_OR_RETURN(mem_case.device_cuda_mem().device_id(), device_id);
    } else {
      OF_UNIMPLEMENTED();
    }
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> CheckOutputBlobObjectsMemCase(LocalCallOpKernelPhyInstrOperand* operand,
                                                          const vm::Stream& stream) {
    DeviceType device_type = JUST(DeviceType4DeviceTag(stream.stream_type().device_tag()));
    const auto& mem_case = JUST(GetMemCase(operand));
    JUST(CheckMemCase(mem_case, device_type, stream.device_id()));
    JUST(operand->ForEachOutputTensor([&](vm::EagerBlobObject* blob_object) -> Maybe<void> {
      CHECK_OR_RETURN(static_cast<bool>(blob_object));
      if (operand->opkernel().need_check_mem_case()) {
        JUST(CheckMemCase(blob_object->mem_case(), device_type, stream.device_id()));
      }
      return Maybe<void>::Ok();
    }));
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> InitOutputBlobs(LocalCallOpKernelPhyInstrOperand* operand) {
    JUST(operand->ForEachOutputTensor([&](vm::EagerBlobObject* blob_object) -> Maybe<void> {
      CHECK_OR_RETURN(static_cast<bool>(blob_object));
      JUST(blob_object->TryInitBlob());
      return Maybe<void>::Ok();
    }));
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> InferTempStorageBlobDesc(LocalCallOpKernelPhyInstrOperand* operand) {
    const auto& InferTmpSizeFn = operand->opkernel().GetInferTmpSizeFn(operand->user_opkernel());
    auto* temp_blob_desc = operand->mut_opkernel()->mut_temp_blob_object()->mut_blob_desc();
    CHECK_OR_RETURN(temp_blob_desc->data_type() == DataType::kChar);
    one::LocalUserOpInferContext* op_infer_ctx =
        operand->opkernel().op_infer_ctx_for_scheduler_thread();
    op_infer_ctx->Update(operand->inputs(), operand->outputs(),
                         operand->consistent_tensor_infer_result());
    size_t temp_size = InferTmpSizeFn(op_infer_ctx);
    temp_blob_desc->mut_shape() = Shape({static_cast<int64_t>(temp_size)});
    temp_blob_desc->set_is_dynamic(true);
    op_infer_ctx->Update(nullptr, nullptr, nullptr);
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> ResetTempStorageBlob(LocalCallOpKernelPhyInstrOperand* operand) {
    JUST(operand->mut_opkernel()->mut_temp_blob_object()->InitBlob());
    return Maybe<void>::Ok();
  }

  template<typename CallbackT>
  static inline Maybe<void> WithComputeContext(LocalCallOpKernelPhyInstrOperand* operand,
                                               DeviceCtx* device_ctx, const CallbackT& Callback) {
    auto* opkernel = operand->mut_opkernel();
    JUST(Callback(opkernel->UpdateComputeContext(operand->inputs(), operand->outputs(),
                                                 operand->consistent_tensor_infer_result(),
                                                 device_ctx)));
    // tensor tuples are not allowed to be hold by StatefulLocalOpKernel
    opkernel->UpdateComputeContext(nullptr, nullptr, nullptr, nullptr);
    return Maybe<void>::Ok();
  }

  static inline void TryInitOpKernelState(LocalCallOpKernelPhyInstrOperand* operand,
                                          DeviceCtx* device_ctx, user_op::OpKernelState** state) {
    if (operand->op_interp_ctx().state) {
      *state = operand->op_interp_ctx().state.get();
      return;
    }
    operand->mut_opkernel()->TryInitOpKernelState(operand->user_opkernel(), device_ctx,
                                                  operand->inputs(), operand->outputs(),
                                                  operand->consistent_tensor_infer_result(), state);
  }

  static inline Maybe<void> AllocateOutputBlobsMemory(LocalCallOpKernelPhyInstrOperand* operand,
                                                      DeviceCtx* device_ctx) {
    Global<one::DTRTensorPool>::Get()->set_current_op_type_name(operand->opkernel().op_type_name());
    JUST(operand->ForEachOutputTensor([&](vm::EagerBlobObject* blob_object) -> Maybe<void> {
      JUST(blob_object->TryAllocateBlobBodyMemory(device_ctx));
      return Maybe<void>::Ok();
    }));
    Global<one::DTRTensorPool>::Get()->set_current_op_type_name("");

    if (DTREnabled()) { JUST(CheckOutputInMemory(operand)); }
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> TryAllocateTempStorageBlobMemory(
      LocalCallOpKernelPhyInstrOperand* operand, DeviceCtx* device_ctx) {
    JUST(operand->mut_opkernel()->mut_temp_blob_object()->TryAllocateBlobBodyMemory(device_ctx));
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> OpKernelCompute(LocalCallOpKernelPhyInstrOperand* operand,
                                            DeviceCtx* device_ctx, user_op::OpKernelState* state) {
    JUST(WithComputeContext(operand, device_ctx,
                            [&](user_op::KernelComputeContext* compute_ctx) -> Maybe<void> {
                              operand->user_opkernel()->Compute(compute_ctx, state);
                              return Maybe<void>::Ok();
                            }));

    if (DTREnabled()) {
      for (int i : operand->opkernel().input_tuple_indexes4mut_ibns()) {
        const std::string& op_type_name = operand->opkernel().op_type_name();
        std::cout << "mutable! op: " << op_type_name << ", input " << i;
        std::cout << "set it as non evictable" << std::endl;
        GetDTRInputs(operand)[i]->set_evict_attr(false);
      }
    }
    if (DTRDebugLevel() >= 3) {
      for (int i : operand->opkernel().input_tuple_indexes4mut_ibns()) {
        const auto& mut_input = operand->inputs()->at(i);
        if (mut_input->mem_case().has_device_cuda_mem()) {
          size_t bytes = mut_input->blob_desc().ByteSizeOfBlobBody();
          std::vector<float> tmp(bytes / 4);
          cudaMemcpy(tmp.data(), mut_input->blob().dptr(), bytes,
                     cudaMemcpyKind::cudaMemcpyDeviceToHost);
          float x = 0;
          for (float f : tmp) { x += f; }
          mut_input->hash_ = x;
          mut_input->backup_data_.resize(bytes / 4);
          memcpy(mut_input->backup_data_.data(), tmp.data(), bytes);
          std::cout << ", gpu memory." << std::endl;
        } else {
          std::cout << ", non gpu memory." << std::endl;
        }
      }

      // compare_input_hash flag
      bool compare_input_hash = false;
      for (const auto& base_class_output : *operand->outputs()) {
        if (base_class_output->mem_case().has_device_cuda_mem()) {
          size_t bytes = base_class_output->blob_desc().ByteSizeOfBlobBody();
          CHECK_EQ_OR_RETURN(bytes % 4, 0);
          std::vector<float> tmp(bytes / 4);
          cudaMemcpy(tmp.data(), base_class_output->blob().dptr(), bytes,
                     cudaMemcpyKind::cudaMemcpyDeviceToHost);
          float x = 0;
          for (float f : tmp) { x += f; }
          if (const auto output =
                  std::dynamic_pointer_cast<DTREagerBlobObject>(base_class_output)) {
            if (output->hash_ != -1) {
              if (output->hash_ != x) {
                std::cout << "wrong!!!!"
                          << " compute op: "
                          << output->compute_op()->shared_opkernel()->user_op_conf_->op_type_name()
                          << ", old hash: " << output->hash_ << ", new hash: " << x
                          << ", old data[0]: " << output->backup_data_[0]
                          << ", new data[0]: " << tmp[0]
                          << ", shape: " << output->blob_desc().shape() << std::endl;

                // compare hash of inputs
                compare_input_hash = true;
              } else {
                std::cout << "correct :)"
                          << " compute op: "
                          << output->compute_op()->shared_opkernel()->user_op_conf_->op_type_name()
                          << ", old hash: " << output->hash_ << ", new hash: " << x << std::endl;
              }
            } else {
              std::cout << "first! set hash to " << x << std::endl;
            }
          }
          base_class_output->hash_ = x;
          base_class_output->backup_data_.resize(bytes / 4);
          memcpy(base_class_output->backup_data_.data(), tmp.data(), bytes);
        } else {
          std::cout << "compute non gpu memory, op is: " << operand->opkernel().op_type_name()
                    << std::endl;
        }
      }
      if (compare_input_hash) {
        for (const auto& base_class_input : *operand->inputs()) {
          if (const auto input = std::dynamic_pointer_cast<DTREagerBlobObject>(base_class_input)) {
            if (input->mem_case().has_device_cuda_mem()) {
              size_t bytes = input->blob_desc().ByteSizeOfBlobBody();
              CHECK_EQ_OR_RETURN(bytes % 4, 0);
              std::vector<float> tmp(bytes / 4);
              cudaMemcpy(tmp.data(), input->blob().dptr(), bytes,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost);
              float x = 0;
              for (float f : tmp) { x += f; }
              if (input->hash_ != -1) {
                if (input->hash_ != x) {
                  std::cout << "input hash wrong!!!!"
                            << ", old hash: " << input->hash_ << ", new hash: " << x
                            << ", old data[0]: " << input->backup_data_[0]
                            << ", new data[0]: " << tmp[0]
                            << ", shape: " << input->blob_desc().shape() << std::endl;
                } else {
                  std::cout << "input hash correct :)"
                            << ", shape: " << input->blob_desc().shape() << std::endl;
                }
              } else {
                std::cout << "input not initialized!!!!!" << x << std::endl;
              }
            } else {
              std::cout << "input non gpu memory, op is: " << operand->opkernel().op_type_name()
                        << std::endl;
            }
          }
        }
      }
    }
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> DeallocateTempStorageBlobMemory(
      LocalCallOpKernelPhyInstrOperand* operand, DeviceCtx* device_ctx) {
    JUST(operand->mut_opkernel()->mut_temp_blob_object()->DeallocateBlobDataPtr());
    return Maybe<void>::Ok();
  }
};

bool IsInplace(const DTREagerBlobObjectList& inputs, const DTREagerBlobObjectList& outputs) {
  bool is_in_place = false;
  for (const auto& input : inputs) {
    for (const auto& output : outputs) {
      if (input->object_dptr() == output->object_dptr()) {
        is_in_place = true;
        break;
      }
    }
    if (is_in_place) { break; }
  }
  return is_in_place;
}

Maybe<void> _IncReferenceNumOfRecomputedTensor(
    const std::shared_ptr<vm::LocalCallOpKernelPhyInstrOperand>& operand, int& pinned_num) {
  if (DTRDebugEnabled()) {
    LOG(INFO) << operand.get() << " with type " << operand->shared_opkernel()->op_type_name()
              << " start";
  }
  for (auto& input : GetDTRInputs(operand)) {
    input->pin();
    const auto& dtr_op = input->compute_op();
    if (!input->is_in_memory()) {
      const auto local_call_op = DTROp2LocalCallOp(input->compute_op());
      CHECK_NOTNULL_OR_RETURN(local_call_op);
      one::DTRTensorPool* dtr_pool = Global<one::DTRTensorPool>::Get();
      if (!input->is_bp_required()) { dtr_pool->need_eager_eviction_ebos_.insert(input.get()); }

      if (dtr_pool->operand_visited_.count(input->compute_op()) == 0) {
        dtr_pool->operand_visited_.insert(input->compute_op());

        if (DTRDebugEnabled()) {
          LOG(INFO) << input << " with compute op " << dtr_op << ", type "
                    << dtr_op->shared_opkernel()->op_type_name()
                    << " is not in memory, searching parents..";
        }

        JUST(_IncReferenceNumOfRecomputedTensor(local_call_op, pinned_num));
      }
    } else {
      pinned_num++;
      if (DTRDebugEnabled()) {
        LOG(INFO) << "pin: compute op of " << input << " is " << dtr_op << " with type "
                  << dtr_op->shared_opkernel()->op_type_name();
      }
    }
  }
  if (DTRDebugEnabled()) {
    LOG(INFO) << operand.get() << " with type " << operand->shared_opkernel()->op_type_name()
              << " end";
  }
  return Maybe<void>::Ok();
}

Maybe<int> IncReferenceNumOfRecomputedTensor(
    const std::shared_ptr<vm::LocalCallOpKernelPhyInstrOperand>& operand) {
  int pinned_num = 0;
  JUST(_IncReferenceNumOfRecomputedTensor(operand, pinned_num));
  Global<one::DTRTensorPool>::Get()->operand_visited_.clear();
  return pinned_num;
}

static Maybe<double> GetEstimatedComputeTime(
    const std::shared_ptr<vm::LocalCallOpKernelPhyInstrOperand>& operand) {
  const auto& inputs = *operand->inputs();
  const auto& outputs = *operand->outputs();
  size_t estimated_compute_time = 0;
  for (const auto& input : inputs) { estimated_compute_time += input->BlobBodyBytes(); }
  for (const auto& output : outputs) { estimated_compute_time += output->BlobBodyBytes(); }
  return estimated_compute_time;
}

Maybe<void> _RecursivelyCompute(
    const std::shared_ptr<vm::LocalCallOpKernelPhyInstrOperand>& operand,
    DeviceCtx* device_ctx) {
  // PinGuard guard(operand->inputs());
  if (DTRDebugLevel() >= 2) {
    if (auto* thread_safe_allocator =
            dynamic_cast<vm::ThreadSafeAllocator*>(device_ctx->mut_allocator())) {
      if (auto* dtr_allocator =
              dynamic_cast<vm::DtrCudaAllocator*>(thread_safe_allocator->backend_allocator())) {
        LOG(INFO) << "allocator stats:";
        dtr_allocator->DisplayAllPieces();
      } else {
        CHECK_OR_RETURN(false);
      }
    }
  }

  const auto& inputs = GetDTRInputs(operand);
  const auto& outputs = GetDTROutputs(operand);

  for (auto& output : outputs) {
    output->pin();
    if (DTRDebugEnabled()) {
      LOG(INFO) << "going to (re)compute " << output->compute_op() << "("
                << output->compute_op_type_name() << ") for " << output;
    }
  }

  for (auto& input : inputs) {
    if (!input->is_in_memory()) {
      CHECK_GT_OR_RETURN(input->input_size(), 0);
      // TODO: recursive recompute the inputs
      auto local_call_op = DTROp2LocalCallOp(input->compute_op());
      CHECK_NOTNULL_OR_RETURN(local_call_op);

      if (DTRDebugEnabled()) {
        LOG(INFO) << "going to recompute " << input->compute_op() << "("
                  << input->compute_op_type_name() << ") for " << input << ", whose dptr is "
                  << input->blob().dptr() << ", is in memory: " << input->is_in_memory()
                  << std::endl;
      }
      // TODO for each ptr rather than shared_ptr
      JUST(_RecursivelyCompute(local_call_op, device_ctx));
      Global<one::DTRTensorPool>::Get()->add_recompute_times();
    }
  }

  JUST(CheckInputInMemory(operand.get()));

  JUST(LocalCallOpKernelUtil::ComputeOperand(operand.get(), device_ctx));
  const double compute_time = JUST(GetEstimatedComputeTime(operand));

  JUST(CheckOutputInMemory(operand.get()));
  // update output tensor dtr attrs
  for (auto& output : outputs) {
    output->set_compute_time(compute_time);
    CHECK_GE_OR_RETURN(output->compute_time(), 0);
    output->reset_pesudo_node();
    Global<one::DTRTensorPool>::Get()->update_after_compute(output.get());
  }

  // unpin output
  if (DTRDebugEnabled()) { LOG(INFO) << "unpin output"; }
  for (auto& output : outputs) { output->unpin(); }
  if (DTRDebugEnabled()) { LOG(INFO) << "unpin output end"; }

  // use current timestamp as access time and **then** update timestamp
  for (auto& input : inputs) { input->update_access_time(); }
  for (auto& output : outputs) { output->update_access_time(); }

  // eager eviction in remat
  for (auto& input : inputs) {
    input->unpin();
    if (input->num_pinned() == 0
        && Global<one::DTRTensorPool>::Get()->need_eager_eviction_ebos_.count(input.get()) > 0) {
      if (DTRDebugEnabled()) {
        LOG(INFO) << "going to evict " << input << " in recomputation, whose dptr is "
                  << input->blob().dptr() << ", compute op: " << input->compute_op_type_name()
                  << ", size: " << input->blob().ByteSizeOfBlobBody()
                  << ", is in memory: " << input->is_in_memory() << std::endl;
      }
      Global<one::DTRTensorPool>::Get()->need_eager_eviction_ebos_.erase(input.get());
      JUST(input->evict());
    }
  }

  // update timestamp
  Global<one::DTRTensorPool>::Get()->time_flies(compute_time);
  return Maybe<void>::Ok();
}

Maybe<void> RecursivelyCompute(
    const std::shared_ptr<vm::LocalCallOpKernelPhyInstrOperand>& operand,
    DeviceCtx* device_ctx) {
  int pinned_num = JUST(IncReferenceNumOfRecomputedTensor(operand));
  if (DTRDebugEnabled()) {
    LOG(INFO) << "pinning input tensors ended, pinned num: " << pinned_num;
  }
  JUST(_RecursivelyCompute(operand, device_ctx));
  CHECK_OR_RETURN(Global<one::DTRTensorPool>::Get()->need_eager_eviction_ebos_.empty());
  if (DTRDebugLevel() >= 1) {
    LOG(INFO) << "all compute ok for " << operand->opkernel().op_type_name() << std::endl;
  }
  return Maybe<void>::Ok();
}

Maybe<void> DTRComputeInstruction(vm::Instruction* instruction) {
  auto operand = JUST(GetLocalCallOpKernelPhyInstrOperand(instruction));
  DeviceCtx* device_ctx = instruction->stream().device_ctx().get();

  if (DTRDebugEnabled()) {
    LOG(INFO) << "all compute start for " << operand->opkernel().op_type_name() << std::endl;
    LOG(INFO) << "start pinning input tensors..";
  }

  const auto& inputs = GetDTRInputs(operand);
  const auto& outputs = GetDTROutputs(operand);

  for (auto& output : outputs) { output->set_compute_op(operand); }
  for (auto& input : inputs) { input->AppendUserOp(operand); }
  JUST(RecursivelyCompute(operand, device_ctx));
  bool is_inplace = IsInplace(inputs, outputs);
  for (const auto& output : outputs) {
    if (is_inplace) {
      output->set_evict_attr(false);
    } else {
      JUST(Global<one::DTRTensorPool>::Get()->insert(output));
    }
  }
  if (DTRDebugLevel() >= 3) { JUST(Global<one::DTRTensorPool>::Get()->display2()); }
  return Maybe<void>::Ok();
}

inline Maybe<void> LocalCallOpKernelUtil::ComputeInstruction(vm::Instruction* instruction) {
  auto operand = JUST(GetLocalCallOpKernelPhyInstrOperand(instruction));
  DeviceCtx* device_ctx = instruction->stream().device_ctx().get();
  return ComputeOperand(operand.get(), device_ctx);
}

Maybe<void> DTRUtil::recompute(vm::DTREagerBlobObject* object, DeviceCtx* device_ctx) {
  if (object->is_in_memory()) { return Maybe<void>::Ok(); }
  return RecursivelyCompute(DTROp2LocalCallOp(object->compute_op()), device_ctx);
}

void LocalCallOpKernelInstructionType::Infer(vm::Instruction* instruction) const {
  UNIMPLEMENTED();
}

void LocalCallOpKernelInstructionType::Compute(vm::Instruction* instruction) const {
  if (DTREnabled()) {
    CHECK_OK(DTRComputeInstruction(instruction));
  } else {
    CHECK_OK(LocalCallOpKernelUtil::ComputeInstruction(instruction));
  }
}

const std::string& LocalCallOpKernelInstructionType::DebugOpTypeName(
    vm::Instruction* instruction) const {
  auto operand = CHECK_JUST(GetLocalCallOpKernelPhyInstrOperand(instruction));
  return operand->opkernel().op_type_name();
}

}  // namespace vm
}  // namespace oneflow
