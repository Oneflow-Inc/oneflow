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

#include "oneflow/core/eager/op_call_phy_instr_operand.h"
#include "oneflow/core/eager/lazy_job_phy_instr_operand.h"
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

namespace oneflow {

namespace one {
class StatefulOpKernel;
class TensorTuple;
class MirroredTensor;
class ConsistentTensorInferResult;
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
  Maybe<void> LaunchLazyJob(const one::EagerBlobObjectListPtr& inputs,
                            const one::EagerBlobObjectListPtr& outputs,
                            const one::EagerBlobObjectListPtr& parameters,
                            const std::shared_ptr<NNGraphIf>& nn_graph);

  // soft sync for inputs/outputs buffers of NNGraph
  Maybe<void> SoftSyncNNGraphBuffers(const one::EagerBlobObjectListPtr& eager_blob_objects,
                                     const std::shared_ptr<NNGraphIf>& nn_graph);

  Maybe<JobDesc> GetJobConfSymbol(const JobConfigProto& job_conf);

  Maybe<ParallelDesc> GetParallelDescSymbol(const ParallelConf& parallel_conf);

  Maybe<Scope> GetScopeSymbol(const ScopeProto& scope_proto);

  Maybe<OperatorConfSymbol> GetOpConfSymbol(const OperatorConf& op_conf);

  Maybe<void> ReleaseTensor(const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object);

  Maybe<void> TouchTensors(const one::EagerBlobObjectListPtr& eager_blob_object);

  template<typename T>
  Maybe<void> SyncAccessBlobByCallback(const T tensor, const std::shared_ptr<BlockingThenBusy>& btb,
                                       const std::function<void(uint64_t)>& Callback,
                                       const std::string& modifier);

  template<typename T>
  Maybe<void> AccessBlobByCallback(const T tensor, const std::function<void(uint64_t)>& callback,
                                   const std::string& modifier);

  Maybe<void> GlobalSync();
  Maybe<void> Barrier(const std::function<void()>& callback);

  Maybe<Scope> BuildInitialScope(int64_t session_id, const JobConfigProto& job_conf,
                                 const std::string& device_tag,
                                 const std::vector<std::string>& machine_device_ids,
                                 const std::shared_ptr<Shape>& hierarchy, bool is_mirrored);

  Maybe<Scope> BuildInitialScopeWithPlacement(int64_t session_id, const JobConfigProto& job_conf,
                                              Symbol<ParallelDesc> placement, bool is_mirrored);

  Maybe<Scope> BuildScopeWithNewParallelDesc(const std::shared_ptr<Scope>& scope,
                                             const std::string& device_tag,
                                             const std::vector<std::string>& machine_device_ids,
                                             const std::shared_ptr<Shape>& hierarchy);

  Maybe<Scope> BuildScopeWithNewParallelConf(const std::shared_ptr<Scope>& scope,
                                             const ParallelConf& parallel_conf);

  Maybe<Scope> BuildScopeWithNewIsMirrored(const std::shared_ptr<Scope>& scope, bool is_mirrored);

  Maybe<Scope> BuildScopeWithNewScopeName(const std::shared_ptr<Scope>& scope,
                                          const std::string& scope_name);

  Maybe<Scope> BuildScopeByProtoSetter(
      const std::shared_ptr<Scope>& scope,
      const std::function<void(const std::shared_ptr<ScopeProto>&)>& Setter);

  Maybe<Scope> BuildScopeByProtoStrSetter(
      const std::shared_ptr<Scope>& scope,
      const std::function<std::string(const std::string&)>& StrSetter);

  Maybe<void> Call(const std::shared_ptr<one::StatefulOpKernel>& opkernel,
                   const one::EagerBlobObjectListPtr& input_eager_blob_objects,
                   const one::EagerBlobObjectListPtr& output_eager_blob_objects,
                   const one::OpExprInterpContext& ctx, Symbol<Stream> stream);

  Maybe<void> Call(
      const std::shared_ptr<one::StatefulOpKernel>& opkernel,
      const one::EagerBlobObjectListPtr& input_eager_blob_objects,
      const one::EagerBlobObjectListPtr& output_eager_blob_objects,
      const std::shared_ptr<const one::ConsistentTensorInferResult>& consistent_tensor_infer_result,
      const one::OpExprInterpContext& ctx, Symbol<Stream> stream);

 private:
  Maybe<void> SoftSyncStream(const one::EagerBlobObjectListPtr& eager_blob_objects,
                             Symbol<Stream> stream);
  Maybe<void> SoftSyncStream(
      std::vector<intrusive::shared_ptr<LocalDepObject>>&& compute_local_dep_objects,
      const std::string& modifier, Symbol<Stream> stream);

 private:
  template<typename PhyInstrOperandT>
  Maybe<void> MakeCriticalSectionBegin(vm::Stream* vm_stream,
                                       const std::shared_ptr<PhyInstrOperandT>& phy_instr_operand);

  template<typename PhyInstrOperandT>
  Maybe<void> MakeCriticalSectionEnd(vm::Stream* vm_stream,
                                     const std::shared_ptr<PhyInstrOperandT>& phy_instr_operand);

  vm::InstructionList* instruction_list_;
};

// Make VM instructions with instruction builder and run instructions with physical/local view.
Maybe<void> PhysicalRun(const std::function<Maybe<void>(InstructionsBuilder*)>& Build);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_INSTRUCTIONS_BUILDER_H_
