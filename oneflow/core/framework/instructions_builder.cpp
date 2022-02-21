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
#include <atomic>
#include "oneflow/core/common/multi_client.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/device/event_record.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/scope.cfg.h"
#include "oneflow/core/framework/parallel_conf_util.h"
#include "oneflow/core/framework/object_storage.h"
#include "oneflow/core/operator/op_node_signature.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/operator/interface_blob_conf.cfg.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/vm/no_arg_cb_phy_instr_operand.h"
#include "oneflow/core/vm/access_blob_arg_cb_phy_instr_operand.h"
#include "oneflow/core/vm/consume_local_dep_object_phy_instr_operand.h"
#include "oneflow/core/eager/release_tensor_arg_phy_instr_operand.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/framework/consistent_tensor_infer_cache.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/job/env_desc.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/vm/tensor_view_operand.h"
#include "oneflow/core/platform/include/pthread_fork.h"

namespace oneflow {

namespace {

Maybe<Symbol<Device>> RawGetCriticalSectionDevice() { return Device::New("critical_section"); }

static constexpr auto* GetCriticalSectionDevice =
    DECORATE(&RawGetCriticalSectionDevice, ThreadLocal);

}  // namespace

template<typename PhyInstrOperandT>
Maybe<void> InstructionsBuilder::MakeCriticalSectionBegin(
    const std::shared_ptr<PhyInstrOperandT>& phy_instr_operand) {
  auto instruction = intrusive::make_shared<vm::InstructionMsg>(
      Global<VirtualMachine>::Get()->mut_vm(), "CriticalSectionBegin",
      std::shared_ptr<const ParallelDesc>(), phy_instr_operand);
  instruction_list_->EmplaceBack(std::move(instruction));
  return Maybe<void>::Ok();
}

template<typename PhyInstrOperandT>
Maybe<void> InstructionsBuilder::MakeCriticalSectionEnd(
    const std::shared_ptr<PhyInstrOperandT>& phy_instr_operand) {
  auto instruction = intrusive::make_shared<vm::InstructionMsg>(
      Global<VirtualMachine>::Get()->mut_vm(), "CriticalSectionEnd",
      std::shared_ptr<const ParallelDesc>(), phy_instr_operand);
  instruction_list_->EmplaceBack(std::move(instruction));
  return Maybe<void>::Ok();
}

// clang-format off
// Job e.g.:
//                                    [wait_and_send_ids]
//                                             |
//                                             V
//                                             |
//                         +-------------------+
//                         |                   |
//                         V             [cpu_decoder]
//                         |                   |
//             [critcial_section_wait]         V
//                         |                   |
//                         V            [forward_ops...]
//                         |                   |
//                         |                   V
//                         +-------------------+
//                                             |
//                                        [copy_loss]
//                                             |
//                                             +-----------------------+
//                                             |                       |
//                                             V                       V
//                                             |                       |
//                                     [backward_ops...]               |
//                                             |                       |
//                                             V            [critical_section_callback]
//                                             |                       |
//                                     [optimizer_ops...]              V
//                                             |                       |
//                                             V                       |
//                                             |                       |
//                                             +-----------------------+
//                                             |                       
//                                     [callback_notifier]                       
// 
//
// clang-format on
// critcial_section_wait is a blocking opkernel which waits tick signal from instruction
// CriticalSectionBegin.
// critical_section_callback is a non-blocking opkernel which notifies instruction
// CriticalSectionEnd done.
Maybe<void> InstructionsBuilder::LaunchLazyJob(const one::EagerBlobObjectListPtr& inputs,
                                               const one::EagerBlobObjectListPtr& outputs,
                                               const one::EagerBlobObjectListPtr& parameters,
                                               const std::shared_ptr<NNGraphIf>& nn_graph) {
  JUST(SoftSyncNNGraphBuffers(inputs, nn_graph));
  JUST(SoftSyncNNGraphBuffers(outputs, nn_graph));
  JUST(SoftSyncNNGraphBuffers(parameters, nn_graph));
  {
    // instruction chain: [CriticalSectionBegin] -> [CriticalSectionEnd]
    // instructions LaunchLazyJob are launched independent from instruction chains
    // [CriticalSectionBegin] -> [CriticalSectionEnd]
    const auto& input_op_name2end_event_record =
        std::make_shared<HashMap<std::string, std::shared_ptr<SharedEventRecord>>>();
    {
      for (const auto& op_name : nn_graph->inputs_op_names()) {
        const auto& event_record = std::make_shared<SharedEventRecord>();
        CHECK_OR_RETURN(input_op_name2end_event_record->emplace(op_name, event_record).second);
      }
      const auto& phy_instr_operand =
          std::make_shared<vm::InputCriticalSectionBeginPhyInstrOperand>(
              nn_graph, inputs, input_op_name2end_event_record);
      JUST(MakeCriticalSectionBegin(phy_instr_operand));
    }
    const auto& output_op_name2end_event_record =
        std::make_shared<HashMap<std::string, std::shared_ptr<SharedEventRecord>>>();
    {
      for (const auto& op_name : nn_graph->outputs_op_names()) {
        const auto& event_record = std::make_shared<SharedEventRecord>();
        CHECK_OR_RETURN(output_op_name2end_event_record->emplace(op_name, event_record).second);
      }
      const auto& phy_instr_operand =
          std::make_shared<vm::OutputCriticalSectionBeginPhyInstrOperand>(
              nn_graph, outputs, output_op_name2end_event_record);
      JUST(MakeCriticalSectionBegin(phy_instr_operand));
    }
    {
      const auto& phy_instr_operand =
          std::make_shared<vm::LaunchLazyJobPhyInstrOperand>(nn_graph, parameters);
      auto instruction = intrusive::make_shared<vm::InstructionMsg>(
          Global<VirtualMachine>::Get()->mut_vm(), "LaunchLazyJob",
          std::shared_ptr<const ParallelDesc>(), phy_instr_operand);
      instruction_list_->EmplaceBack(std::move(instruction));
    }
    for (int i = 0; i < nn_graph->inputs_op_names().size(); ++i) {
      const auto& eager_blob_object = inputs->at(i);
      const auto& op_name = nn_graph->inputs_op_names().at(i);
      const auto& event_record = JUST(MapAt(*input_op_name2end_event_record, op_name));
      const auto& phy_instr_operand = std::make_shared<vm::InputCriticalSecondEndPhyInstrOperand>(
          eager_blob_object, event_record);
      JUST(MakeCriticalSectionEnd(phy_instr_operand));
    }
    for (int i = 0; i < nn_graph->outputs_op_names().size(); ++i) {
      const auto& eager_blob_object = outputs->at(i);
      const auto& op_name = nn_graph->outputs_op_names().at(i);
      const auto& event_record = JUST(MapAt(*output_op_name2end_event_record, op_name));
      const auto& phy_instr_operand = std::make_shared<vm::OutputCriticalSecondEndPhyInstrOperand>(
          eager_blob_object, event_record);
      JUST(MakeCriticalSectionEnd(phy_instr_operand));
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::SoftSyncNNGraphBuffers(
    const one::EagerBlobObjectListPtr& eager_blob_objects,
    const std::shared_ptr<NNGraphIf>& nn_graph) {
  const auto& op_device = JUST(GetCriticalSectionDevice());
  JUST(SoftSyncStream(eager_blob_objects, op_device));
  return Maybe<void>::Ok();
}

Maybe<int64_t> InstructionsBuilder::CreateSymbolId(const cfg::JobConfigProto& job_conf) {
  int64_t symbol_id = JUST(id_generator_->NewSymbolId());
  JUST(AddSymbol<cfg::JobConfigProto, JobConfigProto, JobDesc>(symbol_id, job_conf));
  return symbol_id;
}

Maybe<int64_t> InstructionsBuilder::CreateSymbolId(const cfg::ParallelConf& parallel_conf) {
  int64_t symbol_id = JUST(id_generator_->NewSymbolId());
  JUST(AddSymbol<cfg::ParallelConf, ParallelConf, ParallelDesc>(symbol_id, parallel_conf));
  return symbol_id;
}

Maybe<int64_t> InstructionsBuilder::CreateSymbolId(const cfg::ScopeProto& scope_proto) {
  int64_t symbol_id = JUST(id_generator_->NewSymbolId());
  JUST(AddSymbol<cfg::ScopeProto, ScopeProto, Scope>(symbol_id, scope_proto));
  return symbol_id;
}

Maybe<int64_t> InstructionsBuilder::CreateSymbolId(const cfg::OperatorConf& op_conf) {
  int64_t symbol_id = JUST(id_generator_->NewSymbolId());
  JUST(AddSymbol<cfg::OperatorConf, OperatorConf, OperatorConfSymbol>(symbol_id, op_conf));
  return symbol_id;
}

Maybe<JobDesc> InstructionsBuilder::GetJobConfSymbol(
    const std::shared_ptr<cfg::JobConfigProto>& job_conf) {
  int64_t symbol_id = JUST(FindOrCreateSymbolId(*job_conf));
  return Global<symbol::Storage<JobDesc>>::Get()->MaybeGetPtr(symbol_id);
}

Maybe<ParallelDesc> InstructionsBuilder::GetParallelDescSymbol(
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  int64_t symbol_id = JUST(FindOrCreateSymbolId(*parallel_conf));
  return Global<symbol::Storage<ParallelDesc>>::Get()->MaybeGetPtr(symbol_id);
}

Maybe<Scope> InstructionsBuilder::GetScopeSymbol(
    const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
  int64_t symbol_id = JUST(FindOrCreateSymbolId(*scope_proto));
  return Global<symbol::Storage<Scope>>::Get()->MaybeGetPtr(symbol_id);
}

Maybe<OperatorConfSymbol> InstructionsBuilder::GetOpConfSymbol(
    const std::shared_ptr<cfg::OperatorConf>& op_conf) {
  int64_t symbol_id = JUST(FindOrCreateSymbolId(*op_conf));
  return Global<symbol::Storage<OperatorConfSymbol>>::Get()->MaybeGetPtr(symbol_id);
}

Maybe<Scope> InstructionsBuilder::BuildInitialScope(
    int64_t session_id, const std::shared_ptr<cfg::JobConfigProto>& job_conf,
    const std::string& device_tag, const std::vector<std::string>& machine_device_ids,
    const std::shared_ptr<Shape>& hierarchy, bool is_mirrored) {
  std::shared_ptr<cfg::ScopeProto> scope_proto = std::make_shared<cfg::ScopeProto>();
  scope_proto->set_session_id(session_id);
  std::shared_ptr<JobDesc> job_conf_sym = JUST(GetJobConfSymbol(job_conf));
  scope_proto->set_job_desc_symbol_id(JUST(job_conf_sym->symbol_id()));
  std::shared_ptr<cfg::ParallelConf> parallel_conf =
      JUST(MakeParallelConf(device_tag, machine_device_ids, hierarchy));
  std::shared_ptr<ParallelDesc> device_parallel_desc_sym =
      JUST(GetParallelDescSymbol(parallel_conf));
  scope_proto->set_device_parallel_desc_symbol_id(JUST(device_parallel_desc_sym->symbol_id()));
  parallel_conf = JUST(MakeParallelConf("cpu", machine_device_ids, hierarchy));
  std::shared_ptr<ParallelDesc> host_parallel_desc_sym = JUST(GetParallelDescSymbol(parallel_conf));
  scope_proto->set_host_parallel_desc_symbol_id(JUST(host_parallel_desc_sym->symbol_id()));
  if (is_mirrored) {
    scope_proto->mutable_opt_mirrored_parallel_conf()->mutable_mirrored_parallel();
  } else {
    scope_proto->mutable_opt_mirrored_parallel_conf()->clear_mirrored_parallel();
  }
  return GetScopeSymbol(scope_proto);
}

Maybe<Scope> InstructionsBuilder::BuildScopeWithNewParallelDesc(
    const std::shared_ptr<Scope>& scope, const std::string& device_tag,
    const std::vector<std::string>& machine_device_ids, const std::shared_ptr<Shape>& hierarchy) {
  const auto SetScopeProto = [this, &device_tag, &machine_device_ids,
                              &hierarchy](const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
    std::shared_ptr<cfg::ParallelConf> parallel_conf =
        CHECK_JUST(MakeParallelConf(device_tag, machine_device_ids, hierarchy));
    std::shared_ptr<ParallelDesc> device_parallel_desc_sym =
        CHECK_JUST(GetParallelDescSymbol(parallel_conf));
    parallel_conf = CHECK_JUST(MakeParallelConf("cpu", machine_device_ids, hierarchy));
    std::shared_ptr<ParallelDesc> host_parallel_desc_sym =
        CHECK_JUST(GetParallelDescSymbol(parallel_conf));
    scope_proto->set_device_parallel_desc_symbol_id(
        CHECK_JUST(device_parallel_desc_sym->symbol_id()));
    scope_proto->set_host_parallel_desc_symbol_id(CHECK_JUST(host_parallel_desc_sym->symbol_id()));
  };

  return BuildScopeByProtoSetter(scope, SetScopeProto);
}

Maybe<Scope> InstructionsBuilder::BuildScopeWithNewParallelConf(
    const std::shared_ptr<Scope>& scope, const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  const std::shared_ptr<
      std::tuple<std::string, std::vector<std::string>, std::shared_ptr<cfg::ShapeProto>>>&
      tag_and_dev_ids_and_hierarchy =
          JUST(GetDeviceTagAndMachineDeviceIdsAndHierarchy(parallel_conf));
  std::shared_ptr<Shape> hierarchy;
  if (std::get<2>(*tag_and_dev_ids_and_hierarchy)) {
    ShapeProto hierarchy_proto;
    parallel_conf->hierarchy().ToProto(&hierarchy_proto);
    hierarchy.reset(new Shape(hierarchy_proto));
  }
  return BuildScopeWithNewParallelDesc(scope, std::get<0>(*tag_and_dev_ids_and_hierarchy),
                                       std::get<1>(*tag_and_dev_ids_and_hierarchy), hierarchy);
}

Maybe<Scope> InstructionsBuilder::BuildScopeWithNewIsMirrored(const std::shared_ptr<Scope>& scope,
                                                              bool is_mirrored) {
  const auto SetScopeProto = [is_mirrored](const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
    if (is_mirrored) {
      scope_proto->mutable_opt_mirrored_parallel_conf()->mutable_mirrored_parallel();
    } else {
      scope_proto->mutable_opt_mirrored_parallel_conf()->clear_mirrored_parallel();
    }
  };

  return BuildScopeByProtoSetter(scope, SetScopeProto);
}

Maybe<Scope> InstructionsBuilder::BuildScopeWithNewScopeName(const std::shared_ptr<Scope>& scope,
                                                             std::string scope_name) {
  const auto SetScopeProto = [&scope_name](const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
    scope_proto->add_scope_op_name_prefixes(scope_name);
  };

  return BuildScopeByProtoSetter(scope, SetScopeProto);
}

Maybe<Scope> InstructionsBuilder::BuildScopeByProtoSetter(
    const std::shared_ptr<Scope>& scope,
    const std::function<void(const std::shared_ptr<cfg::ScopeProto>&)>& Setter) {
  std::shared_ptr<cfg::ScopeProto> scope_proto = JUST(scope->MakeChildScopeProto());
  Setter(scope_proto);
  return GetScopeSymbol(scope_proto);
}

Maybe<void> InstructionsBuilder::LocalCallOpKernel(
    const std::shared_ptr<one::StatefulLocalOpKernel>& opkernel,
    const one::EagerBlobObjectListPtr& input_eager_blob_objects,
    const one::EagerBlobObjectListPtr& output_eager_blob_objects,
    const one::OpExprInterpContext& ctx, Symbol<Device> op_device) {
  return LocalCallOpKernel(opkernel, input_eager_blob_objects, output_eager_blob_objects, nullptr,
                           ctx, op_device);
}

Maybe<void> InstructionsBuilder::LocalCallOpKernel(
    const std::shared_ptr<one::StatefulLocalOpKernel>& opkernel,
    const one::EagerBlobObjectListPtr& input_eager_blob_objects,
    const one::EagerBlobObjectListPtr& output_eager_blob_objects,
    const std::shared_ptr<const one::ConsistentTensorInferResult>& consistent_tensor_infer_result,
    const one::OpExprInterpContext& ctx, Symbol<Device> op_device) {
  const auto& parallel_desc_sym = JUST(Placement4Device(op_device)).shared_from_symbol();
  JUST(SoftSyncStream(output_eager_blob_objects, op_device));
  JUST(SoftSyncStream(input_eager_blob_objects, op_device));
  auto phy_instr_operand = JUST(vm::LocalCallOpKernelPhyInstrOperand::New(
      opkernel, input_eager_blob_objects, output_eager_blob_objects, consistent_tensor_infer_result,
      ctx, *one::CurrentDevVmDepObjectConsumeMode()));
  auto instruction = intrusive::make_shared<vm::InstructionMsg>(
      Global<VirtualMachine>::Get()->mut_vm(), JUST(op_device->local_call_instruction_name()),
      parallel_desc_sym, phy_instr_operand);
  instruction_list_->EmplaceBack(std::move(instruction));
  for (const auto& output : *output_eager_blob_objects) {
    if (!output->producer_op_device().has_value()) {
      JUST(output->init_producer_op_device(op_device));
    }
    output->set_last_used_device(op_device);
  }
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::ReleaseTensor(
    const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
    const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  if (pthread_fork::IsForkedSubProcess() && parallel_desc
      && parallel_desc->device_type() != DeviceType::kCPU) {
    return Maybe<void>::Ok();
  }
  const auto& last_used_device = JUST(eager_blob_object->last_used_device());
  const auto& producer_op_device = JUST(eager_blob_object->producer_op_device());
  if (last_used_device != producer_op_device) {
    JUST(SoftSyncStream({JUST(eager_blob_object->compute_local_dep_object())}, "mut",
                        last_used_device));
  }
  Optional<Symbol<Device>> op_device{};
  if (*one::CurrentDevVmDepObjectConsumeMode() == one::DevVmDepObjectConsumeMode::NONE) {
    op_device = Optional<Symbol<Device>>(NullOpt);
  } else if (last_used_device->type() == "async_launched_nccl"
             && (producer_op_device->type() == "cuda" || producer_op_device->type() == "gpu")) {
    // Disable inter-device instruction sequential for tensor used by nccl stream.
    // It's not acceptable for us that cuda compute stream is blocked by cuda nccl stream.
    op_device = Optional<Symbol<Device>>(NullOpt);
  } else if (producer_op_device->type() == "async_launched_nccl") {
    // Disable inter-device instruction sequential for tensor produced by nccl stream.
    op_device = Optional<Symbol<Device>>(NullOpt);
  } else {
    op_device = producer_op_device;
  }
  const auto& phy_instr_operand =
      std::make_shared<vm::ReleaseTensorArgPhyInstrOperand>(eager_blob_object, op_device);
  auto instruction = intrusive::make_shared<vm::InstructionMsg>(
      Global<VirtualMachine>::Get()->mut_vm(), producer_op_device->type() + ".ReleaseTensor",
      parallel_desc, phy_instr_operand);
  instruction_list_->EmplaceBack(std::move(instruction));
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::SoftSyncStream(
    const one::EagerBlobObjectListPtr& eager_blob_objects, Symbol<Device> op_device) {
  SmallSet<Symbol<Device>> last_used_devices;
  for (const auto& eager_blob_object : *eager_blob_objects) {
    const auto& opt_last_used_device = eager_blob_object->last_used_device();
    if (unlikely(!opt_last_used_device.has_value())) { continue; }
    const auto& last_used_device = JUST(opt_last_used_device);
    if (last_used_device != op_device) { SmallSetInsert(&last_used_devices, last_used_device); }
  }
  for (const auto& last_used_device : last_used_devices) {
    std::vector<intrusive::shared_ptr<LocalDepObject>> dep_objects;
    dep_objects.reserve(eager_blob_objects->size());
    for (const auto& eager_blob_object : *eager_blob_objects) {
      const auto& opt_last_used_device = eager_blob_object->last_used_device();
      if (unlikely(!opt_last_used_device.has_value())) { continue; }
      if (JUST(opt_last_used_device) == last_used_device) {
        dep_objects.emplace_back(JUST(eager_blob_object->compute_local_dep_object()));
      }
      eager_blob_object->set_last_used_device(op_device);
    }
    JUST(SoftSyncStream(std::move(dep_objects), "mut", last_used_device));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::SoftSyncStream(
    std::vector<intrusive::shared_ptr<LocalDepObject>>&& compute_local_dep_objects,
    const std::string& modifier, Symbol<Device> op_device) {
  if (!JUST(op_device->need_soft_sync_stream())) { return Maybe<void>::Ok(); }
  OF_PROFILER_RANGE_PUSH("SoftStream");
  const auto& parallel_desc = JUST(Placement4Device(op_device)).shared_from_symbol();
  const auto& phy_instr_operand = std::make_shared<vm::ConsumeLocalDepObjectPhyInstrOperand>(
      std::move(compute_local_dep_objects), modifier);
  auto instruction = intrusive::make_shared<vm::InstructionMsg>(
      Global<VirtualMachine>::Get()->mut_vm(), parallel_desc->device_tag() + ".RecordEvent",
      parallel_desc, phy_instr_operand);
  instruction_list_->EmplaceBack(std::move(instruction));
  OF_PROFILER_RANGE_POP();
  return Maybe<void>::Ok();
}

namespace {

const std::shared_ptr<const ParallelDesc>& GetParallelDesc(
    const std::shared_ptr<one::MirroredTensor> tensor) {
  const auto& device = CHECK_JUST(tensor->device());
  const auto& placement = CHECK_JUST(Placement4Device(device));
  return placement.shared_from_symbol();
}

const std::shared_ptr<const ParallelDesc>& GetParallelDesc(
    const one::EagerMirroredTensorImpl* tensor) {
  const auto& placement = CHECK_JUST(Placement4Device(tensor->device()));
  return placement.shared_from_symbol();
}

}  // namespace

template<typename T>
Maybe<void> InstructionsBuilder::TensorView(const T input_tensor, const T view_tensor) {
  /**
   * TensorView instruction assign the data pointer of input tensor to output view tensor,
   * so they can share memory.
   */
  const auto& parallel_desc = GetParallelDesc(input_tensor);
  const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object =
      JUST(input_tensor->eager_blob_object());
  const std::shared_ptr<vm::EagerBlobObject>& view_eager_blob_object =
      JUST(view_tensor->eager_blob_object());
  // init view blob (with empty data pointer)
  JUST(view_eager_blob_object->InitBlobWithOffset(JUST(view_tensor->storage_offset())));
  view_eager_blob_object->set_is_shape_synced(true);
  view_eager_blob_object->set_last_used_device(JUST(input_tensor->device()));
  // prepare instruction operand
  const auto& phy_instr_operand =
      std::make_shared<vm::TensorViewOperand>(eager_blob_object, view_eager_blob_object);
  // prepare instruction
  auto instruction = intrusive::make_shared<vm::InstructionMsg>(
      Global<VirtualMachine>::Get()->mut_vm(), parallel_desc->device_tag() + ".TensorView",
      parallel_desc, phy_instr_operand);
  // assign the data pointer to output view blob
  instruction_list_->EmplaceBack(std::move(instruction));
  return Maybe<void>::Ok();
}

template Maybe<void> InstructionsBuilder::TensorView(
    const std::shared_ptr<one::MirroredTensor> input_tensor,
    const std::shared_ptr<one::MirroredTensor> view_tensor);

template<typename T>
Maybe<void> InstructionsBuilder::SyncAccessBlobByCallback(
    const T tensor, const std::shared_ptr<BlockingThenBusy>& btb,
    const std::function<void(uint64_t)>& Callback, const std::string& modifier) {
  // We want balance the cpu overhead and notification latency.
  //
  // balanced timeline here:
  //
  //   B: blocking wait
  //   W: wake up
  //   S: spin wait
  //
  //   vm thread:    |<--------------- prev ops ------------------>|<- Callback() ->|
  //
  //   main thread:  |<-------------------- B -------------------->|<- W ->|<- S  ->|
  //
  // bad timeline with more notification latency:
  //
  //   B: blocking wait
  //   W: wake up
  //   S: spin wait
  //
  //   vm thread:    |<--------------- prev ops ------------------>|<- Callback() ->|
  //
  //   main thread:  |<---------------------------- B ----------------------------->|<- W ->|
  //
  // bad timeline with more cpu overhead:
  //
  //   B: blocking wait
  //   W: wake up
  //   S: spin wait
  //
  //   vm thread:    |<--------------- prev ops ------------------>|<- Callback() ->|
  //                 |                                             |                |
  //   main thread:  |<---------------------------- S ----------------------------->|

  const auto& CallbackWrapper = [btb, Callback](uint64_t ofblob_ptr) {
    btb->mut_blocking_counter()->Decrease();
    Callback(ofblob_ptr);
    btb->mut_spin_counter()->Decrease();
  };
  return AccessBlobByCallback(tensor, CallbackWrapper, modifier);
}

template Maybe<void> InstructionsBuilder::SyncAccessBlobByCallback(
    const std::shared_ptr<one::MirroredTensor> tensor, const std::shared_ptr<BlockingThenBusy>& btb,
    const std::function<void(uint64_t)>& Callback, const std::string& modifier);

template Maybe<void> InstructionsBuilder::SyncAccessBlobByCallback(
    const one::EagerMirroredTensorImpl* tensor, const std::shared_ptr<BlockingThenBusy>& btb,
    const std::function<void(uint64_t)>& Callback, const std::string& modifier);

template<typename T>
Maybe<void> InstructionsBuilder::AccessBlobByCallback(const T tensor,
                                                      const std::function<void(uint64_t)>& callback,
                                                      const std::string& modifier) {
  const auto& parallel_desc = GetParallelDesc(tensor);
  const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object = JUST(tensor->eager_blob_object());
  const auto& phy_instr_operand =
      std::make_shared<vm::AccessBlobArgCbPhyInstrOperand>(eager_blob_object, callback, modifier);
  auto instruction = intrusive::make_shared<vm::InstructionMsg>(
      Global<VirtualMachine>::Get()->mut_vm(),
      parallel_desc->device_tag() + ".AccessBlobByCallback", parallel_desc, phy_instr_operand);
  instruction_list_->EmplaceBack(std::move(instruction));
  return Maybe<void>::Ok();
}

template Maybe<void> InstructionsBuilder::AccessBlobByCallback(
    const std::shared_ptr<one::MirroredTensor> tensor,
    const std::function<void(uint64_t)>& callback, const std::string& modifier);

template Maybe<void> InstructionsBuilder::AccessBlobByCallback(
    const one::EagerMirroredTensorImpl* tensor, const std::function<void(uint64_t)>& callback,
    const std::string& modifier);

Maybe<void> InstructionsBuilder::ComputeRankFrontSeqCallback(
    const std::function<void()>& callback) {
  const auto& phy_instr_operand = std::make_shared<vm::NoArgCbPhyInstrOperand>(callback);
  auto instruction = intrusive::make_shared<vm::InstructionMsg>(
      Global<VirtualMachine>::Get()->mut_vm(), "ComputeRankFrontSeqCallback",
      std::shared_ptr<const ParallelDesc>(), phy_instr_operand);
  instruction_list_->PushBack(instruction.Mutable());
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::ComputeGlobalFrontSeqBarrier() {
  const auto& phy_instr_operand = std::make_shared<vm::NoArgCbPhyInstrOperand>([] {});
  auto instruction = intrusive::make_shared<vm::InstructionMsg>(
      Global<VirtualMachine>::Get()->mut_vm(), "ComputeGlobalFrontSeqBarrier",
      std::shared_ptr<const ParallelDesc>(), phy_instr_operand);
  instruction_list_->PushBack(instruction.Mutable());
  return Maybe<void>::Ok();
}

Maybe<void> PhysicalRun(const std::function<Maybe<void>(InstructionsBuilder*)>& Build) {
  vm::InstructionMsgList instruction_list;
  InstructionsBuilder instructions_builder(std::make_shared<vm::PhysicalIdGenerator>(),
                                           &instruction_list);
  JUST(Build(&instructions_builder));
  JUST(vm::Run(instructions_builder.mut_instruction_list()));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
