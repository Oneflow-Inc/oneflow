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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/eager/dtr_util.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/dtr_cuda_allocator.h"
#include "oneflow/core/eager/opkernel_instruction_type.h"
#include "oneflow/core/eager/local_call_opkernel_phy_instr_operand.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/thread_safe_allocator.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/operator/op_node_signature_desc.h"
#include "oneflow/core/operator/op_conf_symbol.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/profiler/collection.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/framework/tensor_pool.h"
#include "oneflow/core/job/env_global_objects_scope.h"

namespace oneflow {
namespace vm {

using DTREagerBlobObjectList = std::vector<std::shared_ptr<vm::DTREagerBlobObject>>;

struct LocalCallOpKernelUtil final {
  static inline Maybe<void> Compute(const vm::InstructionMsg& instr_msg) {
    auto* operand = LocalCallOpKernelUtil::GetLocalCallOpKernelPhyInstrOperand(instr_msg).get();
    DeviceCtx* device_ctx = instr_msg.phy_instr_stream()->device_ctx().get();
    return ComputeOperand(operand, device_ctx);
  }

  static inline Maybe<void> ComputeOperand(LocalCallOpKernelPhyInstrOperand* operand,
                                           DeviceCtx* device_ctx) {
    if (dtr::is_enabled_and_debug()) {
      LOG(INFO) << operand->shared_opkernel()->op_type_name() << " ComputeOperand" << std::endl;
    }
    OF_PROFILER_RANGE_PUSH("ResetPrior");
    operand->mut_opkernel()->composed_attrs_for_scheduler_thread()->ResetPrior(operand->attrs());
    OF_PROFILER_RANGE_POP();
    OF_PROFILER_RANGE_PUSH("AllocateOutputBlobsMemory");
    JUST(AllocateOutputBlobsMemory(operand, device_ctx));
    OF_PROFILER_RANGE_POP();
    if (unlikely(operand->need_temp_storage())) {
      OF_PROFILER_RANGE_GUARD("TryAllocateTempStorageBlobMemory");
      InferTempStorageBlobDesc(operand);
      JUST(TryAllocateTempStorageBlobMemory(operand, device_ctx));
    }
    user_op::OpKernelState* state = nullptr;
    user_op::OpKernelCache* cache = nullptr;
    if (operand->user_opkernel()->has_state_or_cache()) {
      OF_PROFILER_RANGE_GUARD("TryInitOpKernelStateAndCache");
      TryInitOpKernelStateAndCache(operand, device_ctx, &state, &cache);
    }

    if (dtr::is_enabled()) { JUST(CheckInputInMemory(operand)); }
    OpKernelCompute(operand, device_ctx, state, cache);
    if (unlikely(operand->need_temp_storage())) {
      OF_PROFILER_RANGE_GUARD("DeallocateTempStorageBlobMemory");
      JUST(DeallocateTempStorageBlobMemory(operand, device_ctx));
    }
    if (dtr::debug_level() >= 2) {
      LOG(INFO) << operand->shared_opkernel()->op_type_name() << " ComputeOperand done";
    }
    if (dtr::is_enabled()) { JUST(CheckOutputInMemory(operand)); }
    return Maybe<void>::Ok();
  }

  static inline std::shared_ptr<LocalCallOpKernelPhyInstrOperand>
  GetLocalCallOpKernelPhyInstrOperand(const vm::InstructionMsg& instr_msg) {
    auto operand = CHECK_NOTNULL(instr_msg.phy_instr_operand());
    return CHECK_NOTNULL(std::dynamic_pointer_cast<LocalCallOpKernelPhyInstrOperand>(operand));
  }

 private:
  static inline void InferTempStorageBlobDesc(LocalCallOpKernelPhyInstrOperand* operand) {
    const auto& InferTmpSizeFn = operand->opkernel().GetInferTmpSizeFn(operand->user_opkernel());
    auto* temp_eager_blob_object = operand->mut_opkernel()->mut_temp_blob_object();
    CHECK(temp_eager_blob_object->data_type() == DataType::kChar);
    one::LocalUserOpInferContext* op_infer_ctx =
        operand->opkernel().op_infer_ctx_for_scheduler_thread();
    op_infer_ctx->Update(operand->inputs().get(), operand->outputs().get(),
                         operand->consistent_tensor_infer_result().get());
    size_t temp_size = InferTmpSizeFn(op_infer_ctx);
    temp_eager_blob_object->mut_shape() = Shape({static_cast<int64_t>(temp_size)});
    temp_eager_blob_object->mut_stride() = Stride(temp_eager_blob_object->mut_shape());
    temp_eager_blob_object->set_pin_memory(false);
    temp_eager_blob_object->set_is_dynamic(true);
    op_infer_ctx->Update(nullptr, nullptr, nullptr);
  }

  static inline void TryInitOpKernelStateAndCache(LocalCallOpKernelPhyInstrOperand* operand,
                                                  DeviceCtx* device_ctx,
                                                  user_op::OpKernelState** state,
                                                  user_op::OpKernelCache** cache) {
    if (likely(operand->op_interp_ctx().state)) {
      *state = operand->op_interp_ctx().state.get();
      // set state to nullptr so that state initialization in TryInitOpKernelStateAndCache will be
      // skipped.
      state = nullptr;
    }
    operand->mut_opkernel()->TryInitOpKernelStateAndCache(
        operand->user_opkernel(), device_ctx, operand->inputs().get(), operand->outputs().get(),
        operand->consistent_tensor_infer_result().get(), state, cache);
  }

  static inline Maybe<void> AllocateOutputBlobsMemory(LocalCallOpKernelPhyInstrOperand* operand,
                                                      DeviceCtx* device_ctx) {
    Global<dtr::TensorPool>::Get()->set_current_op_type_name(operand->opkernel().op_type_name());

    for (const auto& blob_object : *operand->outputs()) {
      JUST(blob_object->TryAllocateBlobBodyMemory(device_ctx));
    }

    Global<dtr::TensorPool>::Get()->set_current_op_type_name("");

    if (dtr::is_enabled()) { JUST(CheckOutputInMemory(operand)); }
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> TryAllocateTempStorageBlobMemory(
      LocalCallOpKernelPhyInstrOperand* operand, DeviceCtx* device_ctx) {
    return operand->mut_opkernel()->mut_temp_blob_object()->TryAllocateBlobBodyMemory(device_ctx);
  }

  static inline void OpKernelCompute(LocalCallOpKernelPhyInstrOperand* operand,
                                     DeviceCtx* device_ctx, user_op::OpKernelState* state,
                                     const user_op::OpKernelCache* cache) {
    auto* opkernel = operand->mut_opkernel();
    auto* compute_ctx =
        opkernel->UpdateComputeContext(operand->inputs().get(), operand->outputs().get(),
                                       operand->consistent_tensor_infer_result().get(), device_ctx);
    OF_PROFILER_RANGE_PUSH("Compute");
    {
      auto er_guard = CHECK_JUST(profiler::EventRecorder::CreateKernelEventRecorder(
          opkernel->op_type_name(),
#if defined(WITH_CUDA)
          compute_ctx->device_type() == DeviceType::kCUDA
              ? dynamic_cast<ep::CudaStream*>(compute_ctx->stream())->cuda_stream()
              : nullptr,
#endif
          [compute_ctx]() -> std::vector<Shape> {
            std::vector<Shape> shapes;
            for (const auto& pair : compute_ctx->inputs()) {
              shapes.push_back(
                  compute_ctx->TensorDesc4ArgNameAndIndex(pair.first, pair.second)->shape());
            }
            return shapes;
          }));
      operand->user_opkernel()->Compute(compute_ctx, state, cache);
    }
    OF_PROFILER_RANGE_POP();
    // tensor tuples are not allowed to be hold by StatefulLocalOpKernel
    opkernel->UpdateComputeContext(nullptr, nullptr, nullptr, nullptr);

    if (dtr::is_enabled()) {
      for (int i : operand->opkernel().input_tuple_indexes4mut_ibns()) {
        const std::string& op_type_name = operand->opkernel().op_type_name();
        if (dtr::debug_level() >= 3) {
          LOG(INFO) << "mutable! op: " << op_type_name << ", input " << i;
          LOG(INFO) << " set it as non evictable";
        }
        GetDTRInputs(operand)[i]->set_evictable(false);
      }
    }
    if (dtr::is_check_enabled()) {
      for (int i : operand->opkernel().input_tuple_indexes4mut_ibns()) {
        const auto& mut_input = operand->inputs()->at(i);
        if (mut_input->mem_case().has_device_cuda_mem()) {
          size_t bytes = mut_input->ByteSizeOfBlobBody();
          std::vector<float> tmp(bytes / 4);
          cudaMemcpy(tmp.data(), mut_input->dptr(), bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
          float x = 0;
          for (float f : tmp) { x += f; }
          mut_input->hash_ = x;
          mut_input->backup_data_.resize(bytes / 4);
          memcpy(mut_input->backup_data_.data(), tmp.data(), bytes);
        } else {
          LOG(INFO) << "mutable input non gpu memory." << std::endl;
        }
      }

      // compare_input_hash flag
      bool compare_input_hash = false;
      for (const auto& base_class_output : *operand->outputs()) {
        if (base_class_output->mem_case().has_device_cuda_mem()) {
          if (const auto output =
                  std::dynamic_pointer_cast<DTREagerBlobObject>(base_class_output)) {
            size_t bytes = base_class_output->ByteSizeOfBlobBody();
            CHECK_EQ(bytes % 4, 0) << "compute op: " << output->compute_op_type_name();
            std::vector<float> tmp(bytes / 4);
            cudaMemcpy(tmp.data(), base_class_output->dptr(), bytes,
                       cudaMemcpyKind::cudaMemcpyDeviceToHost);
            float x = 0;
            for (float f : tmp) { x += f; }
            if (output->hash_ != -1) {
              if (output->hash_ != x) {
                LOG(INFO) << "wrong!!!!"
                          << " compute op: "
                          << output->compute_op()->shared_opkernel()->user_op_conf_->op_type_name()
                          << ", old hash: " << output->hash_ << ", new hash: " << x
                          << ", old data[0]: " << output->backup_data_[0]
                          << ", new data[0]: " << tmp[0] << ", shape: " << output->shape()
                          << std::endl;

                // compare hash of inputs
                compare_input_hash = true;
              } else {
                LOG(INFO) << "correct :)"
                          << " compute op: "
                          << output->compute_op()->shared_opkernel()->user_op_conf_->op_type_name()
                          << ", old hash: " << output->hash_ << ", new hash: " << x << std::endl;
              }
            } else {
              LOG(INFO) << "first! set " << base_class_output.get() << " hash to " << x
                        << std::endl;
            }
            base_class_output->hash_ = x;
            base_class_output->backup_data_.resize(bytes / 4);
            memcpy(base_class_output->backup_data_.data(), tmp.data(), bytes);
          }
        } else {
          LOG(INFO) << "compute non gpu memory, op is: " << operand->opkernel().op_type_name()
                    << std::endl;
        }
      }
      if (compare_input_hash) {
        for (const auto& base_class_input : *operand->inputs()) {
          if (const auto input = std::dynamic_pointer_cast<DTREagerBlobObject>(base_class_input)) {
            if (input->mem_case().has_device_cuda_mem()) {
              size_t bytes = input->ByteSizeOfBlobBody();
              CHECK_EQ(bytes % 4, 0);
              std::vector<float> tmp(bytes / 4);
              cudaMemcpy(tmp.data(), input->dptr(), bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
              float x = 0;
              for (float f : tmp) { x += f; }
              if (input->hash_ != -1) {
                if (input->hash_ != x) {
                  LOG(INFO) << "input hash wrong!!!!"
                            << ", old hash: " << input->hash_ << ", new hash: " << x
                            << ", old data[0]: " << input->backup_data_[0]
                            << ", new data[0]: " << tmp[0] << ", shape: " << input->shape()
                            << std::endl;
                } else {
                  LOG(INFO) << "input hash correct :)"
                            << ", shape: " << input->shape() << std::endl;
                }
              } else {
                LOG(INFO) << "input not initialized!!!!!" << x << std::endl;
              }
            } else {
              LOG(INFO) << "input non gpu memory, op is: " << operand->opkernel().op_type_name()
                        << std::endl;
            }
          }
        }
      }
    }
  }

  static inline Maybe<void> DeallocateTempStorageBlobMemory(
      LocalCallOpKernelPhyInstrOperand* operand, DeviceCtx* device_ctx) {
    return operand->mut_opkernel()->mut_temp_blob_object()->DeallocateBlobDataPtr();
  }
};

Maybe<void> DTRComputeInstruction(const vm::InstructionMsg& instr_msg);

void LocalCallOpKernelInstructionType::Compute(vm::Instruction* instruction) const {
  if (dtr::is_enabled()) {
    CHECK_JUST(DTRComputeInstruction(instruction->instr_msg()));
  } else {
    CHECK_JUST(LocalCallOpKernelUtil::Compute(instruction->instr_msg()));
  }
}

void LocalCallOpKernelInstructionType::ComputeInFuseMode(vm::InstructionMsg* instr_msg) const {
  if (dtr::is_enabled()) {
    CHECK_OK(DTRComputeInstruction(*instr_msg));
  } else {
    CHECK_JUST(LocalCallOpKernelUtil::Compute(*instr_msg));
  }
}

std::string LocalCallOpKernelInstructionType::DebugOpTypeName(
    const vm::InstructionMsg& instr_msg) const {
  auto* operand = CHECK_NOTNULL(instr_msg.phy_instr_operand().get());
  return CHECK_NOTNULL(dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand))
      ->opkernel()
      .op_type_name();
}

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
  for (auto& input : GetDTRInputs(operand)) {
    input->pin();
    const auto& dtr_op = input->compute_op();
    if (!input->is_in_memory()) {
      const auto local_call_op = DTROp2LocalCallOp(input->compute_op());
      CHECK_NOTNULL_OR_RETURN(local_call_op);
      dtr::TensorPool* dtr_pool = Global<dtr::TensorPool>::Get();
      if (!input->is_bp_required()) { dtr_pool->need_eager_eviction_ebos_.insert(input.get()); }

      if (dtr_pool->operand_visited_.count(input->compute_op()) == 0) {
        dtr_pool->operand_visited_.insert(input->compute_op());

        JUST(_IncReferenceNumOfRecomputedTensor(local_call_op, pinned_num));
      }
    } else {
      pinned_num++;
      if (dtr::debug_level() >= 2) {
        LOG(INFO) << "pin: compute op of " << input << " is " << dtr_op << " with type "
                  << dtr_op->shared_opkernel()->op_type_name();
      }
    }
  }
  if (dtr::debug_level() >= 2) {
    LOG(INFO) << operand << " with type " << operand->shared_opkernel()->op_type_name() << " end";
  }
  return Maybe<void>::Ok();
}

Maybe<int> IncReferenceNumOfRecomputedTensor(
    const std::shared_ptr<vm::LocalCallOpKernelPhyInstrOperand>& operand) {
  int pinned_num = 0;
  JUST(_IncReferenceNumOfRecomputedTensor(operand, pinned_num));
  Global<dtr::TensorPool>::Get()->operand_visited_.clear();
  return pinned_num;
}

static Maybe<double> GetEstimatedComputeTime(vm::LocalCallOpKernelPhyInstrOperand* operand) {
  const auto& inputs = *operand->inputs();
  const auto& outputs = *operand->outputs();
  size_t estimated_compute_time = 0;
  for (const auto& input : inputs) {
    estimated_compute_time += input->tensor_storage()->blob_bytes();
  }
  for (const auto& output : outputs) {
    estimated_compute_time += output->tensor_storage()->blob_bytes();
  }
  return estimated_compute_time;
}

Maybe<void> _RecursivelyCompute(
    const std::shared_ptr<vm::LocalCallOpKernelPhyInstrOperand>& operand, DeviceCtx* device_ctx) {
  // PinGuard guard(operand->inputs());
  if (dtr::debug_level() >= 4) {
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
  }

  for (auto& input : inputs) {
    if (!input->is_in_memory()) {
      CHECK_GT_OR_RETURN(input->input_size(), 0);
      // TODO: recursive recompute the inputs
      auto local_call_op = DTROp2LocalCallOp(input->compute_op());
      CHECK_NOTNULL_OR_RETURN(local_call_op);

      if (dtr::is_enabled_and_debug()) {
        LOG(INFO) << "going to recompute (No." << Global<dtr::TensorPool>::Get()->recompute_times()
                  << ") " << input->compute_op() << "(" << input->compute_op_type_name() << ") for "
                  << input.get() << "(id: " << input->id() << "), whose dptr is " << input->dptr()
                  << ", is in memory: " << input->is_in_memory() << std::endl;
      }
      // TODO for each ptr rather than shared_ptr
      JUST(_RecursivelyCompute(local_call_op, device_ctx));
      Global<dtr::TensorPool>::Get()->add_recompute_times();
    }
  }

  JUST(CheckInputInMemory(operand.get()));

  JUST(LocalCallOpKernelUtil::ComputeOperand(operand.get(), device_ctx));
  const double compute_time = JUST(GetEstimatedComputeTime(operand.get()));

  JUST(CheckOutputInMemory(operand.get()));
  // update output tensor dtr attrs
  for (auto& output : outputs) {
    output->set_compute_time(compute_time);
    CHECK_GE_OR_RETURN(output->compute_time(), 0);
    output->reset_pesudo_node();
    Global<dtr::TensorPool>::Get()->update_after_compute(output.get());
  }

  // unpin output
  if (dtr::debug_level() >= 3) { LOG(INFO) << "unpin output"; }
  for (auto& output : outputs) { output->unpin(); }
  if (dtr::debug_level() >= 3) { LOG(INFO) << "unpin output end"; }

  // use current timestamp as access time and **then** update timestamp
  for (auto& input : inputs) { input->update_access_time(); }
  for (auto& output : outputs) { output->update_access_time(); }

  // eager eviction in remat
  for (auto& input : inputs) {
    input->unpin();
    if (input->num_pinned() == 0
        && Global<dtr::TensorPool>::Get()->need_eager_eviction_ebos_.count(input.get()) > 0) {
      if (dtr::debug_level() >= 2) {
        LOG(INFO) << "going to evict " << input << " in recomputation, whose dptr is "
                  << input->dptr() << ", compute op: " << input->compute_op_type_name()
                  << ", size: " << input->ByteSizeOfBlobBody()
                  << ", is in memory: " << input->is_in_memory() << std::endl;
      }
      Global<dtr::TensorPool>::Get()->need_eager_eviction_ebos_.erase(input.get());
      JUST(input->evict(true));
    }
  }

  // update timestamp
  Global<dtr::TensorPool>::Get()->time_flies(compute_time);
  return Maybe<void>::Ok();
}

Maybe<void> RecursivelyCompute(const std::shared_ptr<vm::LocalCallOpKernelPhyInstrOperand>& operand,
                               DeviceCtx* device_ctx) {
  if (dtr::is_enabled_and_debug()) {
    LOG(INFO) << operand->shared_opkernel()->op_type_name() << " run from user" << std::endl;
  }

  int pinned_num = JUST(IncReferenceNumOfRecomputedTensor(operand));
  if (dtr::debug_level() >= 3) {
    LOG(INFO) << "pinning input tensors ended, pinned num: " << pinned_num;
  }
  JUST(_RecursivelyCompute(operand, device_ctx));
  CHECK_OR_RETURN(Global<dtr::TensorPool>::Get()->need_eager_eviction_ebos_.empty());
  return Maybe<void>::Ok();
}

Maybe<void> DTRComputeInstruction(const vm::InstructionMsg& instr_msg) {
  auto operand = LocalCallOpKernelUtil::GetLocalCallOpKernelPhyInstrOperand(instr_msg);
  DeviceCtx* device_ctx = instr_msg.phy_instr_stream()->device_ctx().get();

  if (EnvBool<ONEFLOW_DTR_OPERATION_LOG>()) {
    LOG(INFO) << "****"
              << "START-DTRComputeInstruction"
              << "-" << instr_msg.instr_type_name() << std::endl;
    // LOG(INFO) << "****" << "START-DTRComputeInstruction" << "-" <<
    // instr_msg.instr_type_name().substr(0, 3) << std::endl;
    LOG(INFO) << "****"
              << "OP-" << operand->opkernel().op_type_name() << std::endl;
  }

  const auto& inputs = GetDTRInputs(operand);
  const auto& outputs = GetDTROutputs(operand);

  for (auto& output : outputs) { output->set_compute_op(operand.get()); }
  for (auto& input : inputs) { input->AppendUserOp(operand.get()); }
  JUST(RecursivelyCompute(operand, device_ctx));
  bool is_inplace = IsInplace(inputs, outputs);
  for (const auto& output : outputs) {
    if (is_inplace) {
      output->set_evictable(false);
    } else {
      JUST(Global<dtr::TensorPool>::Get()->insert(output));
    }
  }
  if (dtr::debug_level() >= 3) { JUST(Global<dtr::TensorPool>::Get()->verbose_display()); }
  if (EnvBool<ONEFLOW_DTR_OPERATION_LOG>()) {
    LOG(INFO) << "****"
              << "END-DTRComputeInstruction"
              << "-" << instr_msg.instr_type_name() << std::endl;
    // LOG(INFO) << "****" << "END-DTRComputeInstruction" << "-" <<
    // instr_msg.instr_type_name().substr(0, 3) << std::endl;
  }
  return Maybe<void>::Ok();
}

Maybe<void> DTRUtil::recompute(vm::DTREagerBlobObject* object, DeviceCtx* device_ctx) {
  if (object->is_in_memory()) { return Maybe<void>::Ok(); }
  return RecursivelyCompute(DTROp2LocalCallOp(object->compute_op()), device_ctx);
}

}  // namespace vm
}  // namespace oneflow
