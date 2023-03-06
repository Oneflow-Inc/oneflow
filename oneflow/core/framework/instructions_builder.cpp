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
#include <thread>
#include <chrono>
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/stream_guard.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/device/event_record.h"
#include "oneflow/core/framework/parallel_conf_util.h"
#include "oneflow/core/operator/op_node_signature.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/singleton_ptr.h"
#include "oneflow/core/common/env_var/vm.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/vm/access_blob_arg_cb_instruction_policy.h"
#include "oneflow/core/vm/ep_record_event_instruction_policy.h"
#include "oneflow/core/vm/op_call_instruction_policy.h"
#include "oneflow/core/vm/barrier_instruction_policy.h"
#include "oneflow/core/vm/critical_section_instruction_policy.h"
#include "oneflow/core/vm/release_tensor_instruction_policy.h"
#include "oneflow/core/vm/lazy_job_instruction_policy.h"
#include "oneflow/core/vm/global_sync_instruction_policy.h"
#include "oneflow/core/vm/op_call_instruction_policy.h"
#include "oneflow/core/vm/stream_wait_instruction_policy.h"
#include "oneflow/core/vm/stream_record_event_instruction_policy.h"
#include "oneflow/core/vm/stream_wait_event_instruction_policy.h"
#include "oneflow/core/vm/sync_access_instruction_policy.h"
#include "oneflow/core/vm/touch_tensors_instruction_policy.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/framework/global_tensor_infer_cache.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/stream_need_soft_sync.h"
#include "oneflow/core/framework/stream_is_comm_net_stream.h"
#include "oneflow/core/framework/stream_support_stream_wait.h"
#include "oneflow/core/framework/stream_on_independent_thread.h"
#include "oneflow/core/job/env_desc.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/vm/allocate_tensor_instruction_policy.h"

namespace oneflow {

namespace {

Maybe<Symbol<Stream>> RawGetCriticalSectionStream() {
  return Stream::New(JUST(Device::New("cpu")), StreamType::kCriticalSection);
}

static constexpr auto* GetCriticalSectionStream =
    DECORATE(&RawGetCriticalSectionStream, ThreadLocal);

Maybe<Symbol<Stream>> RawGetLazyJobLauncherStream() {
  return Stream::New(JUST(Device::New("cpu")), StreamType::kLazyJobLauncher);
}

static constexpr auto* GetLazyJobLauncherStream =
    DECORATE(&RawGetLazyJobLauncherStream, ThreadLocal);

}  // namespace

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
Maybe<void> InstructionsBuilder::LaunchLazyJob(const vm::EagerBlobObjectListPtr& inputs,
                                               const vm::EagerBlobObjectListPtr& outputs,
                                               const vm::EagerBlobObjectListPtr& parameters,
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
        CHECK_OR_RETURN(input_op_name2end_event_record->emplace(op_name, event_record).second)
            << Error::RuntimeError() << "Duplicate Op name " << op_name;
      }

      auto stream = JUST(GetCriticalSectionStream());
      auto* vm_stream = JUST(Singleton<VirtualMachine>::Get()->GetVmStream(stream));
      auto instruction = intrusive::make_shared<vm::Instruction>(
          vm_stream, std::make_shared<vm::InputCriticalSectionBeginInstructionPolicy>(
                         nn_graph, inputs, input_op_name2end_event_record, vm_stream));
      instruction_list_->EmplaceBack(std::move(instruction));
    }
    const auto& output_op_name2end_event_record =
        std::make_shared<HashMap<std::string, std::shared_ptr<SharedEventRecord>>>();
    {
      for (const auto& op_name : nn_graph->outputs_op_names()) {
        const auto& event_record = std::make_shared<SharedEventRecord>();
        CHECK_OR_RETURN(output_op_name2end_event_record->emplace(op_name, event_record).second)
            << Error::RuntimeError() << "Duplicate Op name " << op_name;
      }
      auto stream = JUST(GetCriticalSectionStream());
      auto* vm_stream = JUST(Singleton<VirtualMachine>::Get()->GetVmStream(stream));
      auto instruction = intrusive::make_shared<vm::Instruction>(
          vm_stream, std::make_shared<vm::OutputCriticalSectionBeginInstructionPolicy>(
                         nn_graph, outputs, output_op_name2end_event_record, vm_stream));
      instruction_list_->EmplaceBack(std::move(instruction));
    }
    {
      auto stream = JUST(GetLazyJobLauncherStream());
      auto* vm_stream = JUST(Singleton<VirtualMachine>::Get()->GetVmStream(stream));
      auto instruction = intrusive::make_shared<vm::Instruction>(
          vm_stream, std::make_shared<vm::LaunchLazyJobInstructionPolicy>(nn_graph, parameters));
      instruction_list_->EmplaceBack(std::move(instruction));
    }
    auto stream = JUST(GetCriticalSectionStream());
    auto* vm_stream = JUST(Singleton<VirtualMachine>::Get()->GetVmStream(stream));
    for (int i = 0; i < nn_graph->inputs_op_names().size(); ++i) {
      const auto& eager_blob_object = inputs->at(i);
      const auto& op_name = nn_graph->inputs_op_names().at(i);
      const auto& event_record = JUST(MapAt(*input_op_name2end_event_record, op_name));
      auto instruction = intrusive::make_shared<vm::Instruction>(
          vm_stream, std::make_shared<vm::InputCriticalSectionEndInstructionPolicy>(
                         eager_blob_object, event_record, vm_stream));
      instruction_list_->EmplaceBack(std::move(instruction));
    }
    for (int i = 0; i < nn_graph->outputs_op_names().size(); ++i) {
      const auto& eager_blob_object = outputs->at(i);
      const auto& op_name = nn_graph->outputs_op_names().at(i);
      const auto& event_record = JUST(MapAt(*output_op_name2end_event_record, op_name));
      auto instruction = intrusive::make_shared<vm::Instruction>(
          vm_stream, std::make_shared<vm::OutputCriticalSectionEndInstructionPolicy>(
                         eager_blob_object, event_record, vm_stream));
      instruction_list_->EmplaceBack(std::move(instruction));
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::SoftSyncNNGraphBuffers(
    const vm::EagerBlobObjectListPtr& eager_blob_objects,
    const std::shared_ptr<NNGraphIf>& nn_graph) {
  const auto& stream = JUST(GetCriticalSectionStream());
  JUST(SoftSyncStream(*eager_blob_objects, stream));
  return Maybe<void>::Ok();
}

namespace {

int64_t NewSymbolId() {
  static std::atomic<int64_t> cnt(0);
  return cnt.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

Maybe<JobDesc> InstructionsBuilder::GetJobConfSymbol(const JobConfigProto& job_conf) {
  return Singleton<symbol::Storage<JobDesc>>::Get()->FindOrCreate(job_conf, &NewSymbolId);
}

Maybe<ParallelDesc> InstructionsBuilder::GetParallelDescSymbol(const ParallelConf& parallel_conf) {
  return Singleton<symbol::Storage<ParallelDesc>>::Get()->FindOrCreate(parallel_conf, &NewSymbolId);
}

Maybe<Scope> InstructionsBuilder::GetScopeSymbol(const ScopeProto& scope_proto) {
  return Singleton<symbol::Storage<Scope>>::Get()->FindOrCreate(scope_proto, &NewSymbolId);
}

Maybe<OperatorConfSymbol> InstructionsBuilder::GetOpConfSymbol(const OperatorConf& op_conf) {
  return Singleton<symbol::Storage<OperatorConfSymbol>>::Get()->FindOrCreate(op_conf, &NewSymbolId);
}

Maybe<Scope> InstructionsBuilder::BuildInitialScope(
    int64_t session_id, const JobConfigProto& job_conf, const std::string& device_tag,
    const std::vector<std::string>& machine_device_ids, const std::shared_ptr<Shape>& hierarchy,
    bool is_local) {
  ScopeProto scope_proto;
  scope_proto.set_session_id(session_id);
  std::shared_ptr<JobDesc> job_conf_sym = JUST(GetJobConfSymbol(job_conf));
  scope_proto.set_job_desc_symbol_id(JUST(job_conf_sym->symbol_id()));
  std::shared_ptr<ParallelConf> parallel_conf =
      JUST(MakeParallelConf(device_tag, machine_device_ids, hierarchy));
  std::shared_ptr<ParallelDesc> device_parallel_desc_sym =
      JUST(GetParallelDescSymbol(*parallel_conf));
  scope_proto.set_device_parallel_desc_symbol_id(JUST(device_parallel_desc_sym->symbol_id()));
  parallel_conf = JUST(MakeParallelConf("cpu", machine_device_ids, hierarchy));
  std::shared_ptr<ParallelDesc> host_parallel_desc_sym =
      JUST(GetParallelDescSymbol(*parallel_conf));
  scope_proto.set_host_parallel_desc_symbol_id(JUST(host_parallel_desc_sym->symbol_id()));
  if (is_local) {
    scope_proto.mutable_opt_local_parallel_conf()->mutable_local_parallel();
  } else {
    scope_proto.mutable_opt_local_parallel_conf()->clear_local_parallel();
  }
  return GetScopeSymbol(scope_proto);
}

Maybe<Scope> InstructionsBuilder::BuildInitialScopeWithPlacement(int64_t session_id,
                                                                 const JobConfigProto& job_conf,
                                                                 Symbol<ParallelDesc> placement,
                                                                 bool is_local) {
  ScopeProto scope_proto;
  scope_proto.set_session_id(session_id);
  std::shared_ptr<JobDesc> job_conf_sym = JUST(GetJobConfSymbol(job_conf));
  scope_proto.set_job_desc_symbol_id(JUST(job_conf_sym->symbol_id()));

  std::shared_ptr<ParallelDesc> device_parallel_desc_sym =
      JUST(GetParallelDescSymbol(placement->parallel_conf()));
  scope_proto.set_device_parallel_desc_symbol_id(JUST(device_parallel_desc_sym->symbol_id()));

  Symbol<ParallelDesc> new_placement = JUST(ReplaceDeviceType(placement, DeviceType::kCPU));
  std::shared_ptr<ParallelDesc> host_parallel_desc_sym =
      JUST(GetParallelDescSymbol(new_placement->parallel_conf()));
  scope_proto.set_host_parallel_desc_symbol_id(JUST(host_parallel_desc_sym->symbol_id()));
  if (is_local) {
    scope_proto.mutable_opt_local_parallel_conf()->mutable_local_parallel();
  } else {
    scope_proto.mutable_opt_local_parallel_conf()->clear_local_parallel();
  }
  return GetScopeSymbol(scope_proto);
}

Maybe<Scope> InstructionsBuilder::BuildScopeWithNewParallelDesc(
    const std::shared_ptr<Scope>& scope, const std::string& device_tag,
    const std::vector<std::string>& machine_device_ids, const std::shared_ptr<Shape>& hierarchy) {
  const auto SetScopeProto = [this, &device_tag, &machine_device_ids,
                              &hierarchy](const std::shared_ptr<ScopeProto>& scope_proto) {
    std::shared_ptr<ParallelConf> parallel_conf =
        CHECK_JUST(MakeParallelConf(device_tag, machine_device_ids, hierarchy));
    std::shared_ptr<ParallelDesc> device_parallel_desc_sym =
        CHECK_JUST(GetParallelDescSymbol(*parallel_conf));
    parallel_conf = CHECK_JUST(MakeParallelConf("cpu", machine_device_ids, hierarchy));
    std::shared_ptr<ParallelDesc> host_parallel_desc_sym =
        CHECK_JUST(GetParallelDescSymbol(*parallel_conf));
    scope_proto->set_device_parallel_desc_symbol_id(
        CHECK_JUST(device_parallel_desc_sym->symbol_id()));
    scope_proto->set_host_parallel_desc_symbol_id(CHECK_JUST(host_parallel_desc_sym->symbol_id()));
  };

  return BuildScopeByProtoSetter(scope, SetScopeProto);
}

Maybe<Scope> InstructionsBuilder::BuildScopeWithNewParallelConf(const std::shared_ptr<Scope>& scope,
                                                                const ParallelConf& parallel_conf) {
  const std::shared_ptr<std::tuple<std::string, std::vector<std::string>,
                                   std::shared_ptr<ShapeProto>>>& tag_and_dev_ids_and_hierarchy =
      JUST(GetDeviceTagAndMachineDeviceIdsAndHierarchy(parallel_conf));
  std::shared_ptr<Shape> hierarchy;
  if (std::get<2>(*tag_and_dev_ids_and_hierarchy)) {
    hierarchy.reset(new Shape(parallel_conf.hierarchy()));
  }
  return BuildScopeWithNewParallelDesc(scope, std::get<0>(*tag_and_dev_ids_and_hierarchy),
                                       std::get<1>(*tag_and_dev_ids_and_hierarchy), hierarchy);
}

Maybe<Scope> InstructionsBuilder::BuildScopeWithNewIsLocal(const std::shared_ptr<Scope>& scope,
                                                           bool is_local) {
  const auto SetScopeProto = [is_local](const std::shared_ptr<ScopeProto>& scope_proto) {
    if (is_local) {
      scope_proto->mutable_opt_local_parallel_conf()->mutable_local_parallel();
    } else {
      scope_proto->mutable_opt_local_parallel_conf()->clear_local_parallel();
    }
  };

  return BuildScopeByProtoSetter(scope, SetScopeProto);
}

Maybe<Scope> InstructionsBuilder::BuildScopeWithNewScopeName(const std::shared_ptr<Scope>& scope,
                                                             const std::string& scope_name) {
  const auto SetScopeProto = [&scope_name](const std::shared_ptr<ScopeProto>& scope_proto) {
    scope_proto->add_scope_op_name_prefixes(scope_name);
  };

  return BuildScopeByProtoSetter(scope, SetScopeProto);
}

Maybe<Scope> InstructionsBuilder::BuildScopeByProtoSetter(
    const std::shared_ptr<Scope>& scope,
    const std::function<void(const std::shared_ptr<ScopeProto>&)>& Setter) {
  std::shared_ptr<ScopeProto> scope_proto = JUST(scope->MakeChildScopeProto());
  Setter(scope_proto);
  return GetScopeSymbol(*scope_proto);
}

Maybe<Scope> InstructionsBuilder::BuildScopeByProtoStrSetter(
    const std::shared_ptr<Scope>& scope,
    const std::function<std::string(const std::string&)>& StrSetter) {
  std::shared_ptr<ScopeProto> scope_proto = JUST(scope->MakeChildScopeProto());
  std::string serialized_scope_proto = PbMessage2TxtString(*scope_proto);
  std::string new_serialized_scope_proto = StrSetter(serialized_scope_proto);
  CHECK_OR_RETURN(TxtString2PbMessage(new_serialized_scope_proto, scope_proto.get()))
      << Error::RuntimeError() << "scope_proto parse failed";
  return GetScopeSymbol(*scope_proto);
}

Maybe<void> InstructionsBuilder::Call(const std::shared_ptr<one::StatefulOpKernel>& opkernel,
                                      vm::EagerBlobObjectList&& input_eager_blob_objects,
                                      vm::EagerBlobObjectList&& output_eager_blob_objects,
                                      const one::OpExprInterpContext& ctx, Symbol<Stream> stream) {
  return Call(opkernel, std::move(input_eager_blob_objects), std::move(output_eager_blob_objects),
              nullptr, ctx, stream);
}

Maybe<void> InstructionsBuilder::AllocateTensors(const vm::EagerBlobObjectList& eager_blob_objects,
                                                 Symbol<Stream> stream) {
  // try soft sync eager blob objects which have memory allocated.
  JUST(SoftSyncStream(eager_blob_objects, stream));
  auto* vm_stream = JUST(Singleton<VirtualMachine>::Get()->GetVmStream(stream));
  const auto& instruction_policy =
      std::make_shared<vm::AllocateTensorInstructionPolicy>(eager_blob_objects, vm_stream);
  auto instruction = intrusive::make_shared<vm::Instruction>(vm_stream, instruction_policy);
  instruction_list_->EmplaceBack(std::move(instruction));
  for (const auto& eager_blob_object : eager_blob_objects) {
    if (!eager_blob_object->producer_stream().has_value()) {
      JUST(eager_blob_object->init_producer_stream(stream));
    }
    eager_blob_object->set_last_used_stream(stream);
  }
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::Call(
    const std::shared_ptr<one::StatefulOpKernel>& opkernel,
    vm::EagerBlobObjectList&& input_eager_blob_objects,
    vm::EagerBlobObjectList&& output_eager_blob_objects,
    const std::shared_ptr<const one::GlobalTensorInferResult>& global_tensor_infer_result,
    const one::OpExprInterpContext& ctx, Symbol<Stream> stream) {
  stream = JUST(StreamGuard::TryConvertStream(stream));
  Symbol<Stream> allocator_stream = JUST(GetAllocatorStream(stream));
  if (stream != allocator_stream) {
    JUST(AllocateTensors(output_eager_blob_objects, allocator_stream));
  }
  JUST(SoftSyncStream(output_eager_blob_objects, stream));
  JUST(SoftSyncStream(input_eager_blob_objects, stream));
  for (const auto& output : output_eager_blob_objects) {
    if (!output->producer_stream().has_value()) { JUST(output->init_producer_stream(stream)); }
    output->set_last_used_stream(stream);
  }
  auto* vm_stream = JUST(Singleton<VirtualMachine>::Get()->GetVmStream(stream));
  auto instruction = intrusive::make_shared<vm::Instruction>(
      vm_stream, JUST(vm::OpCallInstructionPolicy::New(
                     vm_stream, opkernel, std::move(input_eager_blob_objects),
                     std::move(output_eager_blob_objects), global_tensor_infer_result, ctx,
                     *one::CurrentDevVmDepObjectConsumeMode())));
  instruction_list_->EmplaceBack(std::move(instruction));
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::ReleaseTensor(
    const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
  const auto& last_used_stream = JUST(eager_blob_object->last_used_stream());
  const auto& producer_stream = JUST(eager_blob_object->producer_stream());
  if (pthread_fork::IsForkedSubProcess()
      && producer_stream->device()->enum_type() != DeviceType::kCPU) {
    return Maybe<void>::Ok();
  }
  Optional<Symbol<Stream>> stream{};
  if (*one::CurrentDevVmDepObjectConsumeMode() == one::DevVmDepObjectConsumeMode::NONE) {
    stream = Optional<Symbol<Stream>>(NullOpt);
  } else if (IsCommNetStream::Visit(last_used_stream->stream_type())) {
    // Disable inter-device instruction sequential for tensor used by communicative stream.
    // It's not acceptable for us that cuda compute stream is blocked by cuda nccl stream.
    stream = Optional<Symbol<Stream>>(NullOpt);
  } else if (IsCommNetStream::Visit(producer_stream->stream_type())) {
    // Disable inter-device instruction sequential for tensor produced by communicative stream.
    stream = Optional<Symbol<Stream>>(NullOpt);
  } else {
    stream = producer_stream;
  }
  struct EnableStreamWaitOnReleaseTensor final
      : public StreamTypeVisitor<EnableStreamWaitOnReleaseTensor> {
    static bool VisitCompute() { return true; }
    static bool VisitHost2Device() { return true; }
    static bool VisitDevice2Host() { return true; }
    static bool VisitCcl() { return false; }
    static bool VisitBarrier() { return false; }
    static bool VisitCriticalSection() { return false; }
    static bool VisitLazyJobLauncher() { return false; }
    static bool VisitPinnedCompute() { return VisitCompute(); }
  };
  const auto& EnableStreamWait = [&] {
    if (last_used_stream->device() != producer_stream->device()) { return false; }
    if (last_used_stream->stream_type() == producer_stream->stream_type()) { return true; }
    return EnableStreamWaitOnReleaseTensor::Visit(last_used_stream->stream_type())
           && EnableStreamWaitOnReleaseTensor::Visit(producer_stream->stream_type());
  };
  if (last_used_stream != producer_stream) {
    if (stream.has_value() && EnableStreamWait()) {
      JUST(SoftSyncStreamBetween({JUST(eager_blob_object->compute_local_dep_object())},
                                 last_used_stream, JUST(stream)));
    } else {
      JUST(RecordEvent({JUST(eager_blob_object->compute_local_dep_object())}, last_used_stream));
    }
    eager_blob_object->set_last_used_stream(producer_stream);
  }
  auto vm_stream = stream.map([](Symbol<Stream> stream) -> vm::Stream* {
    return CHECK_JUST(Singleton<VirtualMachine>::Get()->GetVmStream(stream));
  });
  StreamType stream_type = producer_stream->stream_type();
  auto instruction = intrusive::make_shared<vm::Instruction>(
      JUST(Singleton<VirtualMachine>::Get()->GetVmStream(producer_stream)),
      JUST(vm::MakeReleaseTensorInstructionPolicy::Visit(stream_type, eager_blob_object,
                                                         vm_stream)));
  instruction_list_->EmplaceBack(std::move(instruction));

  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::TouchTensors(
    const vm::EagerBlobObjectListPtr& eager_blob_objects) {
  Symbol<Device> device = JUST(Device::New("cpu"));
  Symbol<Stream> stream = JUST(GetDefaultStreamByDevice(device));
  return TouchTensors(eager_blob_objects, stream);
}

Maybe<void> InstructionsBuilder::TouchTensors(const vm::EagerBlobObjectListPtr& eager_blob_objects,
                                              Symbol<Stream> stream) {
  JUST(SoftSyncStream(*eager_blob_objects, stream));
  auto instruction = intrusive::make_shared<vm::Instruction>(
      JUST(Singleton<VirtualMachine>::Get()->GetVmStream(stream)),
      std::make_unique<vm::TouchTensorsInstructionPolicy>(*eager_blob_objects));
  instruction_list_->EmplaceBack(std::move(instruction));
  return Maybe<void>::Ok();
}

namespace {

template<typename T>
using SmallSet = small_vector<T>;

template<typename T>
std::pair<typename SmallSet<T>::iterator, bool> SmallSetInsert(SmallSet<T>* vec, const T& elem) {
  for (auto iter = vec->begin(); iter != vec->end(); ++iter) {
    if (*iter == elem) { return std::make_pair(iter, false); }
  }
  vec->push_back(elem);
  return std::make_pair(vec->end() - 1, true);
}

template<typename DoEachT>
Maybe<void> ForEachEagerBlobObjectsNeedingSoftSync(
    const vm::EagerBlobObjectList& eager_blob_objects, Symbol<Stream> stream,
    const DoEachT& DoEach) {
  if (eager_blob_objects.size() <= kOpArgsReservedSize) {
    for (const auto& eager_blob_object : eager_blob_objects) {
      const auto& opt_last_used_stream = eager_blob_object->last_used_stream();
      if (unlikely(!opt_last_used_stream.has_value())) { continue; }
      const auto& last_used_stream = JUST(opt_last_used_stream);
      if (last_used_stream != stream) {
        small_vector<intrusive::shared_ptr<LocalDepObject>> dep_objects{
            intrusive::shared_ptr<LocalDepObject>(
                JUST(eager_blob_object->compute_local_dep_object()))};
        JUST(DoEach(last_used_stream, std::move(dep_objects)));
      }
    }
  } else {
    SmallSet<Symbol<Stream>> last_used_streams;
    for (const auto& eager_blob_object : eager_blob_objects) {
      const auto& opt_last_used_stream = eager_blob_object->last_used_stream();
      if (unlikely(!opt_last_used_stream.has_value())) { continue; }
      const auto& last_used_stream = JUST(opt_last_used_stream);
      if (last_used_stream != stream) { SmallSetInsert(&last_used_streams, last_used_stream); }
    }
    for (const auto& last_used_stream : last_used_streams) {
      small_vector<intrusive::shared_ptr<LocalDepObject>> dep_objects{};
      for (const auto& eager_blob_object : eager_blob_objects) {
        const auto& opt_stream = eager_blob_object->last_used_stream();
        if (unlikely(!opt_stream.has_value())) { continue; }
        if (JUST(opt_stream) == last_used_stream) {
          dep_objects.emplace_back(JUST(eager_blob_object->compute_local_dep_object()));
        }
      }
      JUST(DoEach(last_used_stream, std::move(dep_objects)));
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> InstructionsBuilder::SoftSyncStream(const vm::EagerBlobObjectList& eager_blob_objects,
                                                Symbol<Stream> stream) {
  JUST(ForEachEagerBlobObjectsNeedingSoftSync(
      eager_blob_objects, stream,
      [&](Symbol<Stream> last_used_stream, auto&& dep_objects) -> Maybe<void> {
        return SoftSyncStreamBetween(std::move(dep_objects), last_used_stream, stream);
      }));
  for (const auto& eager_blob_object : eager_blob_objects) {
    eager_blob_object->set_last_used_stream(stream);
  }
  return Maybe<void>::Ok();
}

namespace {

bool SupportingStreamWait(Symbol<Stream> from_stream, Symbol<Stream> to_stream) {
  if (from_stream->device() == to_stream->device()
      && from_stream->stream_type() == to_stream->stream_type()
      && from_stream->thread_uid() == to_stream->thread_uid()) {
    CHECK(from_stream == to_stream);
  }
  if (unlikely(!ThreadLocalEnvBool<ONEFLOW_VM_ENABLE_STREAM_WAIT>())) { return false; }
  DeviceType from_device_type = from_stream->device()->enum_type();
  DeviceType to_device_type = from_stream->device()->enum_type();
  return from_stream->device() == to_stream->device() && from_device_type == DeviceType::kCUDA
         && StreamSupportStreamWait::Visit(from_stream->stream_type(), from_device_type)
         && StreamSupportStreamWait::Visit(to_stream->stream_type(), to_device_type)
         && !StreamOnIndependentThread::Visit(from_stream->stream_type())
         && !StreamOnIndependentThread::Visit(to_stream->stream_type());
}

}  // namespace

Maybe<void> InstructionsBuilder::SoftSyncStreamBetween(
    small_vector<intrusive::shared_ptr<LocalDepObject>>&& dependences, Symbol<Stream> from_stream,
    Symbol<Stream> to_stream) {
  CHECK(from_stream != to_stream) << "synchronization is unnecessary";
  if (SupportingStreamWait(from_stream, to_stream)) {
    JUST(StreamWait(std::move(dependences), from_stream, to_stream));
  } else {
    JUST(RecordEvent(std::move(dependences), from_stream));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::StreamWait(
    small_vector<intrusive::shared_ptr<LocalDepObject>>&& dependences, Symbol<Stream> from_stream,
    Symbol<Stream> to_stream) {
  auto* from_vm_stream = JUST(Singleton<VirtualMachine>::Get()->GetVmStream(from_stream));
  auto* to_vm_stream = JUST(Singleton<VirtualMachine>::Get()->GetVmStream(to_stream));
  if (from_vm_stream->mut_thread_ctx() != to_vm_stream->mut_thread_ctx()) {
    auto stream_record_event =
        std::make_shared<vm::StreamRecordEventInstructionPolicy>(dependences);
    auto record_instruction =
        intrusive::make_shared<vm::Instruction>(from_vm_stream, stream_record_event);
    instruction_list_->EmplaceBack(std::move(record_instruction));
    auto stream_wait_event =
        std::make_shared<vm::StreamWaitEventInstructionPolicy>(dependences, stream_record_event);
    auto wait_instruction =
        intrusive::make_shared<vm::Instruction>(to_vm_stream, stream_wait_event);
    instruction_list_->EmplaceBack(std::move(wait_instruction));
  } else {
    auto instruction = intrusive::make_shared<vm::Instruction>(
        to_vm_stream, std::make_unique<vm::StreamWaitInstructionPolicy>(
                          std::move(dependences), from_vm_stream, to_vm_stream));
    instruction_list_->EmplaceBack(std::move(instruction));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::RecordEvent(
    small_vector<intrusive::shared_ptr<LocalDepObject>>&& compute_local_dep_objects,
    Symbol<Stream> last_used_stream) {
  DeviceType device_type = last_used_stream->device()->enum_type();
  if (!NeedSoftSync::Visit(last_used_stream->stream_type(), device_type)) {
    return Maybe<void>::Ok();
  }
  std::string modifier = "mut";
  StreamType stream_type = last_used_stream->stream_type();
  auto instruction = intrusive::make_shared<vm::Instruction>(
      JUST(Singleton<VirtualMachine>::Get()->GetVmStream(last_used_stream)),
      JUST(GetRecordEventInstructionPolicy::Visit(stream_type, device_type,
                                                  std::move(compute_local_dep_objects), modifier)));
  instruction_list_->EmplaceBack(std::move(instruction));
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> InstructionsBuilder::SyncAccessBlobByCallback(
    const T tensor, const std::shared_ptr<BlockingThenBusy>& btb,
    const std::function<void(ep::Stream*, const std::shared_ptr<vm::EagerBlobObject>&)>& Callback,
    const std::string& modifier) {
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

  const auto& CallbackWrapper = [btb, Callback](
                                    ep::Stream* stream,
                                    const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
    btb->mut_notifier()->Notify();
    Callback(stream, eager_blob_object);
    btb->mut_spin_counter()->Decrease();
  };
  return AccessBlobByCallback(tensor, CallbackWrapper, modifier);
}

template Maybe<void> InstructionsBuilder::SyncAccessBlobByCallback(
    const std::shared_ptr<one::LocalTensor> tensor, const std::shared_ptr<BlockingThenBusy>& btb,
    const std::function<void(ep::Stream*, const std::shared_ptr<vm::EagerBlobObject>&)>& Callback,
    const std::string& modifier);

template Maybe<void> InstructionsBuilder::SyncAccessBlobByCallback(
    const one::EagerLocalTensorImpl* tensor, const std::shared_ptr<BlockingThenBusy>& btb,
    const std::function<void(ep::Stream*, const std::shared_ptr<vm::EagerBlobObject>&)>& Callback,
    const std::string& modifier);

namespace {

Maybe<Symbol<Device>> GetDevice(const std::shared_ptr<one::LocalTensor>& tensor) {
  return tensor->device();  // return Maybe<Symbol<Device>>
}

Maybe<Symbol<Device>> GetDevice(const one::EagerLocalTensorImpl* tensor) {
  return tensor->device();  // return const Symbol<Device>&
}

template<typename T>
Maybe<Symbol<Stream>> GetAccessStream(const T tensor) {
  Symbol<Device> device = JUST(GetDevice(tensor));
  // Do not use producer_stream or last_used_stream.
  // Bug case when using producer_stream or last_used_stream:
  //
  // ```python
  // tensor = oneflow.ones((1024, 1024, 1024), device='cuda').cpu()
  // ndarray = tensor.numpy() # share memory
  //
  // ```
  // `ndarray` may not be ones because instruction AccessBlobByCallback is prescheduled before
  // oneflow.ones actually finished.
  Symbol<Stream> stream = JUST(GetDefaultStreamByDevice(device));
  return StreamGuard::TryConvertStream(stream);
}

}  // namespace

template<typename T>
Maybe<void> InstructionsBuilder::AccessBlobByCallback(
    const T tensor,
    const std::function<void(ep::Stream*, const std::shared_ptr<vm::EagerBlobObject>&)>& callback,
    const std::string& modifier) {
  const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object = JUST(tensor->eager_blob_object());
  Symbol<Stream> stream = JUST(GetAccessStream(tensor));
  JUST(SoftSyncStream({eager_blob_object}, stream));
  auto instruction = intrusive::make_shared<vm::Instruction>(
      // Never replace `stream` with producer_stream or last_used_stream.
      JUST(Singleton<VirtualMachine>::Get()->GetVmStream(stream)),
      std::make_shared<vm::AccessBlobArgCbInstructionPolicy>(eager_blob_object, callback,
                                                             modifier));
  instruction_list_->EmplaceBack(std::move(instruction));
  return Maybe<void>::Ok();
}

template Maybe<void> InstructionsBuilder::AccessBlobByCallback(
    const std::shared_ptr<one::LocalTensor> tensor,
    const std::function<void(ep::Stream*, const std::shared_ptr<vm::EagerBlobObject>&)>& callback,
    const std::string& modifier);

template Maybe<void> InstructionsBuilder::AccessBlobByCallback(
    const one::EagerLocalTensorImpl* tensor,
    const std::function<void(ep::Stream*, const std::shared_ptr<vm::EagerBlobObject>&)>& callback,
    const std::string& modifier);

namespace {

Maybe<Symbol<Stream>> GetBarrierStream() {
  auto device = JUST(Device::New("cpu"));
  return Stream::New(device, StreamType::kBarrier);
}

}  // namespace

Maybe<void> InstructionsBuilder::GlobalSync() {
  auto stream = JUST(GetBarrierStream());
  auto instruction = intrusive::make_shared<vm::Instruction>(
      JUST(Singleton<VirtualMachine>::Get()->GetVmStream(stream)),
      std::make_shared<vm::GlobalSyncInstructionPolicy>());
  instruction_list_->PushBack(instruction.Mutable());
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::Barrier(const std::function<void()>& Callback) {
  auto stream = JUST(GetBarrierStream());
  auto instruction = intrusive::make_shared<vm::Instruction>(
      JUST(Singleton<VirtualMachine>::Get()->GetVmStream(stream)),
      std::make_shared<vm::BarrierInstructionPolicy>(Callback));
  instruction_list_->PushBack(instruction.Mutable());
  return Maybe<void>::Ok();
}

namespace {

template<typename InstructionPolicyT>
Maybe<vm::Instruction*> MutThreadLocalInstruction(Symbol<Stream> stream) {
  static thread_local std::vector<intrusive::shared_ptr<vm::Instruction>> vec;
  if (unlikely(stream->unique_stream_id() >= vec.size())) {
    vec.resize(stream->unique_stream_id() + 1);
  }
  auto* instruction_ptr = &vec[stream->unique_stream_id()];
  if (static_cast<bool>(*instruction_ptr) && (*instruction_ptr)->ref_cnt() != 1) {
    // This instruction should not be reusd because of being hold by other threads.
    instruction_ptr->Reset();
  }
  if (unlikely(!static_cast<bool>(*instruction_ptr))) {
    *instruction_ptr = intrusive::make_shared<vm::Instruction>(
        JUST(Singleton<VirtualMachine>::Get()->GetVmStream(stream)),
        std::make_shared<InstructionPolicyT>());
  }
  return instruction_ptr->Mutable();
}

}  // namespace

template<typename T, typename InstructionPolicyT>
Maybe<void> SyncAccessSmallMem(char* mem_ptr, size_t bytes, const T tensor) {
  static thread_local vm::InstructionList instruction_list;
  static thread_local InstructionsBuilder instructions_builder(&instruction_list);
  const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object = JUST(tensor->eager_blob_object());
  const Symbol<Stream> stream = JUST(GetAccessStream(tensor));
  if (eager_blob_object->last_used_stream().has_value()
      && stream != JUST(eager_blob_object->last_used_stream())) {
    // Synchronize stream.
    JUST(instructions_builder.SoftSyncStream({eager_blob_object}, stream));
  }
  InstructionPolicyT* instruction_policy = nullptr;
  {
    // Construct instruction.
    auto* instruction = JUST(MutThreadLocalInstruction<InstructionPolicyT>(stream));
    instruction_policy =
        static_cast<InstructionPolicyT*>(instruction->mut_instruction_policy());  // NOLINT
    instruction_policy->Reset(mem_ptr, bytes, eager_blob_object.get());
    instruction_list.PushBack(instruction);
  }
  // Dispatch instructions.
  JUST(vm::Run(&instruction_list));
  {
    // This thread should blocking wait if and only if there is a lot of workload on worker thread.
    // When workload is small, we want better performance by skipping cond_.notify_xxx which costs
    // about 2us to 3us.
    auto* virtual_machine = JUST(SingletonMaybe<VirtualMachine>());
    static constexpr int kSkipBlockingThreshold = 2;
    if (virtual_machine->flying_instruction_cnt() < kSkipBlockingThreshold) {
      // skip pthread_cond_broadcast on worker thread.
      instruction_policy->mut_btb()->mut_notifier()->Notify();
    }
  }
  // wait until done.
  JUST(instruction_policy->mut_btb()->WaitUntilCntEqualZero(
      VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> SyncReadSmallMem(char* mem_ptr, size_t bytes, const T tensor) {
  return SyncAccessSmallMem<T, vm::SyncReadInstructionPolicy>(mem_ptr, bytes, tensor);
}

template Maybe<void> SyncReadSmallMem(char* mem_ptr, size_t bytes,
                                      const std::shared_ptr<one::LocalTensor> tensor);

template Maybe<void> SyncReadSmallMem(char* mem_ptr, size_t bytes,
                                      const one::EagerLocalTensorImpl* tensor);

}  // namespace oneflow
