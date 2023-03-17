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
#ifndef ONEFLOW_CORE_FRAMEWORK_INSTRUCTIONS_BUILDER_H_
#define ONEFLOW_CORE_FRAMEWORK_INSTRUCTIONS_BUILDER_H_

#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/scope.pb.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/blocking_then_busy.h"
#include "oneflow/core/operator/op_conf_symbol.h"
#include "oneflow/core/vm/vm_util.h"

namespace oneflow {

namespace one {
class StatefulOpKernel;
class TensorTuple;
class LocalTensor;
class GlobalTensorInferResult;
}  // namespace one

class NNGraphIf;

class SharedEventRecord;

class InstructionsBuilder : public std::enable_shared_from_this<InstructionsBuilder> {
 public:
  InstructionsBuilder(const InstructionsBuilder&) = delete;
  InstructionsBuilder(InstructionsBuilder&&) = delete;
  explicit InstructionsBuilder(vm::InstructionList* instruction_list)
      : instruction_list_(instruction_list) {}
  ~InstructionsBuilder() { instruction_list_->Clear(); }

  const vm::InstructionList& instruction_list() const { return *instruction_list_; }

  vm::InstructionList* mut_instruction_list() { return instruction_list_; }

  // Build VM execution instructions with NNGraph's inputs/outputs/parameters for NNGraph execution.
  Maybe<void> LaunchLazyJob(const vm::EagerBlobObjectListPtr& inputs,
                            const vm::EagerBlobObjectListPtr& outputs,
                            const vm::EagerBlobObjectListPtr& parameters,
                            const std::shared_ptr<NNGraphIf>& nn_graph);

  // soft sync for inputs/outputs buffers of NNGraph
  Maybe<void> SoftSyncNNGraphBuffers(const vm::EagerBlobObjectListPtr& eager_blob_objects,
                                     const std::shared_ptr<NNGraphIf>& nn_graph);

  Maybe<JobDesc> GetJobConfSymbol(const JobConfigProto& job_conf);

  Maybe<ParallelDesc> GetParallelDescSymbol(const ParallelConf& parallel_conf);

  Maybe<Scope> GetScopeSymbol(const ScopeProto& scope_proto);

  Maybe<OperatorConfSymbol> GetOpConfSymbol(const OperatorConf& op_conf);

  Maybe<void> ReleaseTensor(const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object);

  Maybe<void> TouchTensors(const vm::EagerBlobObjectListPtr& eager_blob_objects);

  Maybe<void> TouchTensors(const vm::EagerBlobObjectListPtr& eager_blob_objects,
                           Symbol<Stream> stream);

  template<typename T>
  Maybe<void> SyncAccessBlobByCallback(
      const T tensor, const std::shared_ptr<BlockingThenBusy>& btb,
      const std::function<void(ep::Stream*, const std::shared_ptr<vm::EagerBlobObject>&)>& Callback,
      const std::string& modifier);

  template<typename T>
  Maybe<void> AccessBlobByCallback(
      const T tensor,
      const std::function<void(ep::Stream*, const std::shared_ptr<vm::EagerBlobObject>&)>& callback,
      const std::string& modifier);

  Maybe<void> GlobalSync();
  Maybe<void> Barrier(const std::function<void()>& callback);

  Maybe<Scope> BuildInitialScope(int64_t session_id, const JobConfigProto& job_conf,
                                 const std::string& device_tag,
                                 const std::vector<std::string>& machine_device_ids,
                                 const std::shared_ptr<Shape>& hierarchy, bool is_local);

  Maybe<Scope> BuildInitialScopeWithPlacement(int64_t session_id, const JobConfigProto& job_conf,
                                              Symbol<ParallelDesc> placement, bool is_local);

  Maybe<Scope> BuildScopeWithNewParallelDesc(const std::shared_ptr<Scope>& scope,
                                             const std::string& device_tag,
                                             const std::vector<std::string>& machine_device_ids,
                                             const std::shared_ptr<Shape>& hierarchy);

  Maybe<Scope> BuildScopeWithNewParallelConf(const std::shared_ptr<Scope>& scope,
                                             const ParallelConf& parallel_conf);

  Maybe<Scope> BuildScopeWithNewIsLocal(const std::shared_ptr<Scope>& scope, bool is_local);

  Maybe<Scope> BuildScopeWithNewScopeName(const std::shared_ptr<Scope>& scope,
                                          const std::string& scope_name);

  Maybe<Scope> BuildScopeByProtoSetter(
      const std::shared_ptr<Scope>& scope,
      const std::function<void(const std::shared_ptr<ScopeProto>&)>& Setter);

  Maybe<Scope> BuildScopeByProtoStrSetter(
      const std::shared_ptr<Scope>& scope,
      const std::function<std::string(const std::string&)>& StrSetter);

  Maybe<void> Call(const std::shared_ptr<one::StatefulOpKernel>& opkernel,
                   vm::EagerBlobObjectList&& input_eager_blob_objects,
                   vm::EagerBlobObjectList&& output_eager_blob_objects,
                   const one::OpExprInterpContext& ctx, Symbol<Stream> stream);

  Maybe<void> Call(
      const std::shared_ptr<one::StatefulOpKernel>& opkernel,
      vm::EagerBlobObjectList&& input_eager_blob_objects,
      vm::EagerBlobObjectList&& output_eager_blob_objects,
      const std::shared_ptr<const one::GlobalTensorInferResult>& global_tensor_infer_result,
      const one::OpExprInterpContext& ctx, Symbol<Stream> stream);

  Maybe<void> SoftSyncStream(const vm::EagerBlobObjectList& eager_blob_objects,
                             Symbol<Stream> stream);

 private:
  Maybe<void> AllocateTensors(const vm::EagerBlobObjectList& eager_blob_objects,
                              Symbol<Stream> stream);

  Maybe<void> SoftSyncStreamBetween(
      small_vector<intrusive::shared_ptr<LocalDepObject>>&& dependences, Symbol<Stream> from_stream,
      Symbol<Stream> to_stream);

  Maybe<void> StreamWait(small_vector<intrusive::shared_ptr<LocalDepObject>>&& dependences,
                         Symbol<Stream> from_stream, Symbol<Stream> to_stream);

  Maybe<void> RecordEvent(
      small_vector<intrusive::shared_ptr<LocalDepObject>>&& compute_local_dep_objects,
      Symbol<Stream> stream);

  vm::InstructionList* instruction_list_;
};

// Make VM instructions with instruction builder and run instructions with physical/local view.
template<typename CallbackT>
Maybe<void> PhysicalRun(const CallbackT& Build) {
  vm::InstructionList instruction_list;
  InstructionsBuilder instructions_builder(&instruction_list);
  JUST(Build(&instructions_builder));
  JUST(vm::Run(instructions_builder.mut_instruction_list()));
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> SyncReadSmallMem(char* mem_ptr, size_t bytes, const T tensor);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_INSTRUCTIONS_BUILDER_H_
