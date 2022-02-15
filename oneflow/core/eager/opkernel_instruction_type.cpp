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
#include "oneflow/core/common/protobuf.h"
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

template<typename DoEachT>
Maybe<void> ForEachDTROutputTensor(LocalCallOpKernelPhyInstrOperand* operand,
                                   const DoEachT& DoEach) {
  auto* ptr = dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand);
  if (!ptr) { return Maybe<void>::Ok(); }
  for (const auto& output : *ptr->outputs()) {
    CHECK_OR_RETURN(static_cast<bool>(output.get()));
    auto shared_dtr_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(output);
    CHECK_NOTNULL_OR_RETURN(shared_dtr_blob_object);
    JUST(DoEach(shared_dtr_blob_object));
  }
  return Maybe<void>::Ok();
}

template<typename DoEachT>
Maybe<void> ForEachDTRInputTensor(LocalCallOpKernelPhyInstrOperand* operand,
                                  const DoEachT& DoEach) {
  auto* ptr = dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand);
  CHECK_NOTNULL_OR_RETURN(ptr);
  for (const auto& input : *ptr->inputs()) {
    CHECK_OR_RETURN(static_cast<bool>(input.get()));
    auto dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(input.get());
    CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
    JUST(DoEach(dtr_blob_object));
  }
  return Maybe<void>::Ok();
}

std::shared_ptr<LocalCallOpKernelPhyInstrOperand> DTROp2LocalCallOp(DTRInstrOperand* operand) {
  const auto& inputs = operand->inputs();
  const auto& outputs = operand->outputs();

  std::shared_ptr<one::EagerBlobObjectList> input_shared_ptr =
      std::make_shared<one::EagerBlobObjectList>(inputs.size());
  std::shared_ptr<one::EagerBlobObjectList> output_shared_ptr =
      std::make_shared<one::EagerBlobObjectList>(outputs.size());

  for (int i = 0; i < inputs.size(); ++i) {
    if (auto input = inputs[i].lock()) {
      input_shared_ptr->at(i) = input;
    } else {
      // CHECK_JUST(Global<one::DTRTensorPool>::Get()->display2());
      LOG(FATAL) << "null at input " << i << " of op " << operand->shared_opkernel()->op_type_name();
    }
  }

  for (int i = 0; i < outputs.size(); ++i) {
    if (auto output = outputs[i].lock()) {
      output_shared_ptr->at(i) = output;
    } else {
      // CHECK_JUST(Global<one::DTRTensorPool>::Get()->display2());
      LOG(FATAL) << "null at output " << i << " of op " << operand->shared_opkernel()->op_type_name();
    }
  }

  auto phy_instr_operand = std::make_shared<LocalCallOpKernelPhyInstrOperand>(
      operand->shared_opkernel(), input_shared_ptr, output_shared_ptr,
      operand->consistent_tensor_infer_result(), operand->op_interp_ctx(),
      operand->dev_vm_dep_object_consume_mode());

  return phy_instr_operand;
}

struct LocalCallOpKernelUtil {
  static Maybe<void> CheckInputInMemory(LocalCallOpKernelPhyInstrOperand* operand) {
    return ForEachDTRInputTensor(operand,
                                 [](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
                                   CHECK_OR_RETURN(dtr_blob_object->is_in_memory());
                                   CHECK_NOTNULL_OR_RETURN(dtr_blob_object->blob().dptr());
                                   return Maybe<void>::Ok();
                                 });
  }

  static Maybe<void> CheckOutputInMemory(LocalCallOpKernelPhyInstrOperand* operand) {
    return ForEachDTROutputTensor(
        operand, [&](const std::shared_ptr<vm::DTREagerBlobObject>& object) -> Maybe<void> {
          CHECK_OR_RETURN(object->is_in_memory());
          CHECK_NOTNULL_OR_RETURN(object->blob().dptr());
          return Maybe<void>::Ok();
        });
  }

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
    if (oneflow::DTREnabled()) { JUST(CheckInputInMemory(operand)); }
    JUST(OpKernelCompute(operand, device_ctx, state));

    JUST(DeallocateTempStorageBlobMemory(operand, device_ctx));
    operand->set_user_opkernel(nullptr);
    if (oneflow::DTRDebugEnabled()) {
      LOG(INFO) << operand->shared_opkernel()->op_type_name() << " ComputeOperand done";
    }
    if (oneflow::DTREnabled()) { JUST(CheckOutputInMemory(operand)); }
    return Maybe<void>::Ok();
  }
  static inline Maybe<void> ComputeInstruction(vm::Instruction* instruction);

  static inline Maybe<LocalCallOpKernelPhyInstrOperand> GetLocalCallOpKernelPhyInstrOperand(
      vm::Instruction* instruction) {
    const auto& operand = instruction->instr_msg().phy_instr_operand();
    CHECK_OR_RETURN(static_cast<bool>(operand));
    auto ptr = std::dynamic_pointer_cast<LocalCallOpKernelPhyInstrOperand>(operand);
    CHECK_NOTNULL_OR_RETURN(ptr);
    return ptr;
  }

 public:
  static inline Maybe<LocalCallOpKernelPhyInstrOperand> GetSharedLocalCallOpKernelPhyInstrOperand(
      vm::Instruction* instruction) {
    const auto& operand = instruction->instr_msg().phy_instr_operand();
    CHECK_OR_RETURN(static_cast<bool>(operand));
    auto local_operand = std::dynamic_pointer_cast<LocalCallOpKernelPhyInstrOperand>(operand);
    CHECK_NOTNULL_OR_RETURN(local_operand);
    return local_operand;
  }

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
      CHECK_NOTNULL_OR_RETURN(blob_object->tensor_buffer()->blob_dptr());
      return Maybe<void>::Ok();
    }));
    Global<one::DTRTensorPool>::Get()->set_current_op_type_name("");

    if (oneflow::DTREnabled()) { JUST(CheckOutputInMemory(operand)); }
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

    if (oneflow::DTRDebugLevel() >= 3) {
      for (int i : operand->opkernel().input_tuple_indexes4mut_ibns()) {
        const std::string& op_type_name = operand->opkernel().op_type_name();
        std::cout << "mutable! op: " << op_type_name << ", input " << i;
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

one::EagerBlobObjectListPtr global_pinned_ebos = nullptr;

struct PinGuard {
  OF_DISALLOW_COPY_AND_MOVE(PinGuard);
  explicit PinGuard(const one::EagerBlobObjectListPtr& ebos)
      : ebos_(ebos), old_ebos_(global_pinned_ebos) {
    if (old_ebos_ != nullptr) {
      for (auto& ebo : *old_ebos_) {
        if (auto dtr_ebo = std::dynamic_pointer_cast<DTREagerBlobObject>(ebo)) {
          dtr_ebo->unpin();
        } else {
          CHECK(false);
        }
      }
    }
    for (auto& ebo : *ebos_) {
      if (auto dtr_ebo = std::dynamic_pointer_cast<DTREagerBlobObject>(ebo)) {
        dtr_ebo->pin();
      } else {
        CHECK(false);
      }
    }
    global_pinned_ebos = ebos_;
  }
  ~PinGuard() {
    if (old_ebos_ != nullptr) {
      for (auto& ebo : *old_ebos_) {
        if (auto dtr_ebo = std::dynamic_pointer_cast<DTREagerBlobObject>(ebo)) {
          dtr_ebo->pin();
        } else {
          CHECK(false);
        }
      }
    }
    for (auto& ebo : *ebos_) {
      if (auto dtr_ebo = std::dynamic_pointer_cast<DTREagerBlobObject>(ebo)) {
        dtr_ebo->unpin();
      } else {
        CHECK(false);
      }
    }
    global_pinned_ebos = old_ebos_;
  }

 private:
  one::EagerBlobObjectListPtr ebos_;
  one::EagerBlobObjectListPtr old_ebos_;
};

struct DTRLocalCallOpKernelUtil final : public LocalCallOpKernelUtil {
  static inline Maybe<void> ComputeOperandWithRecompute(
      const std::shared_ptr<oneflow::vm::LocalCallOpKernelPhyInstrOperand>& operand,
      DeviceCtx* device_ctx) {
    // PinGuard guard(operand->inputs());
    if (oneflow::DTRDebugLevel() >= 2) {
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
    JUST(ForEachDTROutputTensor(
        operand.get(), [&](const std::shared_ptr<vm::DTREagerBlobObject>& object) -> Maybe<void> {
          object->pin();
          if (oneflow::DTRDebugEnabled()) {
            LOG(INFO) << "going to (re)compute " << object->compute_op() << "("
                      << object->compute_op_type_name() << ") for " << object;
          }
          return Maybe<void>::Ok();
        }));
    JUST(ForEachDTRInputTensor(
        operand.get(), [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
          if (!dtr_blob_object->is_in_memory()) {
            CHECK_GT_OR_RETURN(dtr_blob_object->input_size(), 0);
            // TODO: recursive recompute the inputs
            auto local_call_op = DTROp2LocalCallOp(dtr_blob_object->compute_op());
            CHECK_NOTNULL_OR_RETURN(local_call_op);

            if (oneflow::DTRDebugEnabled()) {
              LOG(INFO) << "going to recompute " << dtr_blob_object->compute_op() << "("
                        << dtr_blob_object->compute_op_type_name() << ") for " << dtr_blob_object
                        << ", whose dptr is " << dtr_blob_object->blob().dptr()
                        << ", is in memory: " << dtr_blob_object->is_in_memory() << std::endl;
            }
            // TODO for each ptr rather than shared_ptr
            JUST(ComputeOperandWithRecompute(local_call_op, device_ctx));
            Global<one::DTRTensorPool>::Get()->add_recompute_times();
          }
          return Maybe<void>::Ok();
        }));
    JUST(CheckInputInMemory(operand.get()));

    JUST(ComputeOperand(operand.get(), device_ctx));
    const double compute_time = JUST(GetEstimatedComputeTime(operand));

    JUST(ForEachDTROutputTensor(
        operand.get(), [&](const std::shared_ptr<vm::DTREagerBlobObject>& object) -> Maybe<void> {
          CHECK_OR_RETURN(object->is_in_memory());
          object->set_compute_time(compute_time);
          CHECK_GT_OR_RETURN(object->compute_time(), 0);
          object->reset_pesudo_node();
          Global<one::DTRTensorPool>::Get()->update_after_compute(object.get());
          return Maybe<void>::Ok();
        }));

    if (oneflow::DTRDebugEnabled()) { LOG(INFO) << "unpin output"; }
    JUST(ForEachDTROutputTensor(
        operand.get(), [&](const std::shared_ptr<vm::DTREagerBlobObject>& object) -> Maybe<void> {
          object->unpin();
          return Maybe<void>::Ok();
        }));
    if (oneflow::DTRDebugEnabled()) { LOG(INFO) << "unpin output end"; }

    // use current timestamp as access time and **then** update timestamp
    JUST(ForEachDTRInputTensor(operand.get(),
                               [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
                                 dtr_blob_object->update_access_time();
                                 return Maybe<void>::Ok();
                               }));
    JUST(ForEachDTROutputTensor(
        operand.get(), [&](const std::shared_ptr<vm::DTREagerBlobObject>& object) -> Maybe<void> {
          object->update_access_time();
          return Maybe<void>::Ok();
        }));

    JUST(ForEachDTRInputTensor(
        operand.get(), [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
          dtr_blob_object->unpin();
          if (dtr_blob_object->num_pinned() == 0
              && Global<one::DTRTensorPool>::Get()->need_eager_eviction_ebos_.count(dtr_blob_object)
                     > 0) {
            if (oneflow::DTRDebugEnabled()) {
              LOG(INFO) << "going to evict " << dtr_blob_object
                        << " in recomputation, whose dptr is " << dtr_blob_object->blob().dptr()
                        << ", compute op: " << dtr_blob_object->compute_op_type_name()
                        << ", size: " << dtr_blob_object->blob().ByteSizeOfBlobBody()
                        << ", is in memory: " << dtr_blob_object->is_in_memory() << std::endl;
            }
            Global<one::DTRTensorPool>::Get()->need_eager_eviction_ebos_.erase(dtr_blob_object);
            JUST(dtr_blob_object->evict());
          }
          return Maybe<void>::Ok();
        }));
    Global<one::DTRTensorPool>::Get()->time_flies(compute_time);
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> Prepare(vm::Instruction* instruction) {
    auto operand = JUST(GetSharedLocalCallOpKernelPhyInstrOperand(instruction));
    if (oneflow::DTRDebugEnabled()) {
      std::cout << "prepare start for " << operand->opkernel().op_type_name() << std::endl;
    }

    // JUST(Prepare2(instruction));

    if (oneflow::DTRDebugEnabled()) {
      std::cout << "prepare ok for " << operand->opkernel().op_type_name() << std::endl;
      std::cout << "===============================" << std::endl;
    }

    // for (int i : operand->opkernel().input_tuple_indexes4mut_ibns()) {
    //   const auto& mut_input = operand->inputs()->at(i);
    //   if (const auto dtr_input = std::dynamic_pointer_cast<DTREagerBlobObject>(mut_input)) {
    //     dtr_input->set_evict_attr(false);
    //   }
    // }

    return Maybe<void>::Ok();
  }

  static inline Maybe<void> InitOutputBlobAttrs(vm::Instruction* instruction) {
    auto operand = JUST(GetSharedLocalCallOpKernelPhyInstrOperand(instruction));
    JUST(ForEachDTROutputTensor(
        operand.get(),
        [&](const std::shared_ptr<vm::DTREagerBlobObject>& dtr_blob_object) -> Maybe<void> {
          JUST(dtr_blob_object->InitBlobAttrs(operand));
          return Maybe<void>::Ok();
        }));
    return Maybe<void>::Ok();
  }

  static Maybe<double> GetEstimatedComputeTime(
      const std::shared_ptr<oneflow::vm::LocalCallOpKernelPhyInstrOperand>& operand) {
    size_t estimated_compute_time = 0;
    JUST(ForEachDTRInputTensor(operand.get(),
                               [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
                                 estimated_compute_time += dtr_blob_object->BlobBodyBytes();
                                 return Maybe<void>::Ok();
                               }));
    JUST(ForEachDTROutputTensor(
        operand.get(),
        [&](const std::shared_ptr<vm::DTREagerBlobObject>& dtr_blob_object) -> Maybe<void> {
          estimated_compute_time += dtr_blob_object->BlobBodyBytes();
          return Maybe<void>::Ok();
        }));

    return estimated_compute_time;
  }

  static inline Maybe<void> UpdateTensorInfo(vm::Instruction* instruction) {
    auto operand = JUST(GetSharedLocalCallOpKernelPhyInstrOperand(instruction));

    // find in_place op and do sth
    bool is_in_place = false;
    auto* op = dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand.get());
    CHECK_NOTNULL_OR_RETURN(op);
    for (const auto& input : *op->inputs()) {
      CHECK_OR_RETURN(static_cast<bool>(input.get()));
      auto in_dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(input.get());
      CHECK_NOTNULL_OR_RETURN(in_dtr_blob_object);
      for (const auto& output : *op->outputs()) {
        CHECK_OR_RETURN(static_cast<bool>(output.get()));
        auto out_dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(output.get());
        CHECK_NOTNULL_OR_RETURN(out_dtr_blob_object);
        if (in_dtr_blob_object->object_dptr() == out_dtr_blob_object->object_dptr()) {
          is_in_place = true;
          break;
        }
      }
      if (is_in_place) { break; }
    }

    JUST(ForEachDTROutputTensor(
        operand.get(),
        [&](const std::shared_ptr<vm::DTREagerBlobObject>& dtr_blob_object) -> Maybe<void> {
          if (is_in_place) {
            dtr_blob_object->set_evict_attr(false);
          } else {
            JUST(Global<one::DTRTensorPool>::Get()->insert(dtr_blob_object));
          }
          return Maybe<void>::Ok();
        }));

    if (oneflow::DTRDebugLevel() >= 3) { JUST(Global<one::DTRTensorPool>::Get()->display2()); }

    // Display info of current tensor pool
    // if (oneflow::DTRDebugEnabled()) { JUST(Global<one::DTRTensorPool>::Get()->display()); }

    // // Display output shared_ptr's count
    // std::cout << "======================== Display output dtrblobobject shared_ptr's count
    // ========================" << std::endl; size_t output_id = 0; for (const auto& output :
    // *operand->outputs()) {
    //   std::cout << output_id++ << "th output shared_ptr's count: " << output.use_count() <<
    //   std::endl;
    // }
    return Maybe<void>::Ok();
  }

#if 0
  static inline Maybe<void> recompute(vm::DTREagerBlobObject* object, DeviceCtx* device_ctx) {
    if (oneflow::DTRDebugEnabled()) {
      LOG(INFO) << "going to recompute "
                << object->compute_op()->shared_opkernel()->user_op_conf_->op_type_name() << " for "
                << object << ", whose dptr is " << object->blob().dptr()
                << ", is in memory: " << object->is_in_memory() << std::endl;
    }
    auto local_call_op = DTROp2LocalCallOp(object->compute_op());
    CHECK_NOTNULL_OR_RETURN(local_call_op);

    // pin inputs
    // TODO for each ptr rather than shared_ptr
    auto* operand = local_call_op.get();
    // PinGuard guard(operand->inputs());
    JUST(
        ForEachDTRInputTensor(operand, [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
          dtr_blob_object->pin();
          return Maybe<void>::Ok();
        }));

    // recompute inputs not in memory
    JUST(
        ForEachDTRInputTensor(operand, [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
          if (!dtr_blob_object->is_in_memory()) {
            if (dtr_blob_object->input_size() == 0) { LOG(INFO) << dtr_blob_object; }
            CHECK_GT_OR_RETURN(dtr_blob_object->input_size(), 0);
            JUST(recompute(dtr_blob_object, device_ctx));
          }
          dtr_blob_object->update_access_time();
          return Maybe<void>::Ok();
        }));

    // TODO: execute function, update outputs, if execute failure (OOM), evict()
    // auto* ptr = dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand.get());
    CHECK_NOTNULL_OR_RETURN(operand);
    JUST(DoInfer(operand));
    JUST(DoCompute(operand, device_ctx));

    CHECK_GT_OR_RETURN(object->compute_time(), 0);
    Global<one::DTRTensorPool>::Get()->time_flies(object->compute_time());
    // JUST(Global<one::DTRTensorPool>::Get()->time_flies(GetEstimatedComputeTime(instruction)));
    if (oneflow::DTRDebugEnabled()) {
      // if (oneflow::DTRDebugEnabled() || !object->is_in_memory()) {
      CHECK_OR_RETURN(object->is_in_memory());
    }

    // unpin inputs
    JUST(
        ForEachDTRInputTensor(operand, [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
          dtr_blob_object->unpin();
          return Maybe<void>::Ok();
        }));

    Global<one::DTRTensorPool>::Get()->add_recompute_times();
    Global<one::DTRTensorPool>::Get()->update_after_compute(object);
    return Maybe<void>::Ok();
  }
#endif
};

Maybe<void> IncReferenceNumOfRecomputedTensor(
    const std::shared_ptr<vm::LocalCallOpKernelPhyInstrOperand>& operand, int& pinned_num) {
  if (oneflow::DTRDebugEnabled()) {
    LOG(INFO) << operand.get() << " with type " << operand->shared_opkernel()->op_type_name()
              << " start";
  }
  JUST(ForEachDTRInputTensor(
      operand.get(), [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
        dtr_blob_object->pin();
        const auto& dtr_op = dtr_blob_object->compute_op();
        if (!dtr_blob_object->is_in_memory()) {
          const auto local_call_op = DTROp2LocalCallOp(dtr_blob_object->compute_op());
          CHECK_NOTNULL_OR_RETURN(local_call_op);
          if (!dtr_blob_object->is_bp_required()) {
            Global<one::DTRTensorPool>::Get()->need_eager_eviction_ebos_.insert(dtr_blob_object);
          }

          if (Global<one::DTRTensorPool>::Get()->operand_visited_.count(
                  dtr_blob_object->compute_op())
              == 0) {
            Global<one::DTRTensorPool>::Get()->operand_visited_.insert(
                dtr_blob_object->compute_op());

            if (oneflow::DTRDebugEnabled()) {
              LOG(INFO) << dtr_blob_object << " with compute op " << dtr_op << ", type "
                        << dtr_op->shared_opkernel()->op_type_name()
                        << " is not in memory, searching parents..";
            }

            JUST(IncReferenceNumOfRecomputedTensor(local_call_op, pinned_num));
          }
        } else {
          pinned_num++;
          if (oneflow::DTRDebugEnabled()) {
            LOG(INFO) << "pin: compute op of " << dtr_blob_object << " is " << dtr_op
                      << " with type " << dtr_op->shared_opkernel()->op_type_name();
          }
        }
        return Maybe<void>::Ok();
      }));
  if (oneflow::DTRDebugEnabled()) {
    LOG(INFO) << operand.get() << " with type " << operand->shared_opkernel()->op_type_name()
              << " end";
  }
  return Maybe<void>::Ok();
}

inline Maybe<void> LocalCallOpKernelUtil::ComputeInstruction(vm::Instruction* instruction) {
  auto operand = JUST(GetLocalCallOpKernelPhyInstrOperand(instruction));
  DeviceCtx* device_ctx = instruction->stream().device_ctx().get();
  if (oneflow::DTREnabled()) {
    if (oneflow::DTRDebugEnabled()) {
      LOG(INFO) << "all compute start for " << operand->opkernel().op_type_name() << std::endl;
      LOG(INFO) << "start pinning input tensors..";
    }
    int pinned_num = 0;
    JUST(IncReferenceNumOfRecomputedTensor(operand, pinned_num));
    if (oneflow::DTRDebugEnabled()) {
      LOG(INFO) << "pinning input tensors ended, pinned num: " << pinned_num;
    }
    Global<one::DTRTensorPool>::Get()->operand_visited_.clear();
    JUST(DTRLocalCallOpKernelUtil::Prepare(instruction));
    JUST(DTRLocalCallOpKernelUtil::InitOutputBlobAttrs(instruction));
    JUST(ForEachDTRInputTensor(operand.get(),
                               [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
                                 dtr_blob_object->update_user_ops(operand);
                                 return Maybe<void>::Ok();
                               }));
    JUST(DTRLocalCallOpKernelUtil::ComputeOperandWithRecompute(operand, device_ctx));
    JUST(DTRLocalCallOpKernelUtil::UpdateTensorInfo(instruction));
    auto operand =
        JUST(LocalCallOpKernelUtil::GetSharedLocalCallOpKernelPhyInstrOperand(instruction));

    CHECK_OR_RETURN(Global<one::DTRTensorPool>::Get()->need_eager_eviction_ebos_.empty());
    if (oneflow::DTRDebugLevel() >= 1) {
      LOG(INFO) << "all compute ok for " << operand->opkernel().op_type_name() << std::endl;
    }
    return Maybe<void>::Ok();
  } else {
    return ComputeOperand(operand.get(), device_ctx);
  }
}

Maybe<void> DTRUtil::recompute(vm::DTREagerBlobObject* object, DeviceCtx* device_ctx) {
  if (object->is_in_memory()) { return Maybe<void>::Ok(); }
  return DTRLocalCallOpKernelUtil::ComputeOperandWithRecompute(
      DTROp2LocalCallOp(object->compute_op()), device_ctx);
}

void LocalCallOpKernelInstructionType::Infer(vm::Instruction* instruction) const {
  UNIMPLEMENTED();
}

void LocalCallOpKernelInstructionType::Compute(vm::Instruction* instruction) const {
  CHECK_OK(LocalCallOpKernelUtil::ComputeInstruction(instruction));
}

const std::string& LocalCallOpKernelInstructionType::DebugOpTypeName(
    vm::Instruction* instruction) const {
  auto operand =
      CHECK_JUST(LocalCallOpKernelUtil::GetLocalCallOpKernelPhyInstrOperand(instruction));
  return operand->opkernel().op_type_name();
}

}  // namespace vm
}  // namespace oneflow
