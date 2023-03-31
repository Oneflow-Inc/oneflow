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
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/common/cost_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/job/critical_section_instance.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/job/utils/progress_bar.h"
#include "oneflow/core/job_rewriter/job_completer.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/framework/variable_tensor_mgr.h"

namespace oneflow {

namespace {

Maybe<bool> GetTensorValidInCurRank(const std::shared_ptr<one::Tensor>& tensor) {
  if (tensor->is_global()) {
    const auto& parallel_id = JUST(GetParallelId4CurrentProcessCtx(JUST(tensor->parallel_desc())));
    if (parallel_id->has_value()) {
      return true;
    } else {
      return false;
    }
  } else {
    return true;
  }
}

Maybe<std::string> GetTensorMetaString(const std::shared_ptr<one::Tensor>& tensor) {
  std::string ret = "shape=" + tensor->shape()->ToString() + ", dtype=" + tensor->dtype()->name();
  if (tensor->is_global()) {
    ret += ", placement=" + *JUST(PlacementToString(JUST(tensor->parallel_desc())));
    ret += ", nd_sbp=" + NdSbpToString(JUST(tensor->nd_sbp()));
  } else {
    ret += ", device=" + JUST(tensor->device())->ToString();
  }
  return ret;
}

template<typename T>
Maybe<void> MakeEagerBlobObjectList(vm::EagerBlobObjectList* blob_list, const T& tensor_list) {
  blob_list->reserve(tensor_list.size());
  for (const auto& tensor : tensor_list) {
    CHECK_OR_RETURN(tensor->is_eager())
        << Error::RuntimeError() << "Tensors in nn.Graph should be eager";
    if (tensor->is_global()) {
      blob_list->emplace_back(JUST(JUST(tensor->cur_rank_phy_tensor())->eager_blob_object()));
    } else {
      blob_list->emplace_back(JUST(tensor->eager_blob_object()));
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

NNGraph::~NNGraph() {
  VLOG(1) << "Graph destructor Try to close c nn graph name " << name_ << "." << std::endl;
  CHECK_JUST(Close());
}

Maybe<void> NNGraph::Close() {
  if (!is_closed_) {
    VLOG(1) << "Try to close c nn graph name " << name_ << "." << std::endl;
    CloseRuntimeBuffers();
    runtime_.reset();
    session_ctx_->RemoveGraphFreeEagerTensors(name_);
    VLOG(1) << "Finish close c nn graph name " << name_ << "." << std::endl;

    session_ctx_.reset();
    is_closed_ = true;
  }
  return Maybe<void>::Ok();
}

const std::vector<std::string>& NNGraph::inputs_op_names() const { return inputs_op_names_; }

const std::vector<std::string>& NNGraph::outputs_op_names() const { return outputs_op_names_; }

const std::vector<bool>& NNGraph::inputs_valid() const { return input_tensors_valid_; }

const std::vector<bool>& NNGraph::outputs_valid() const { return output_tensors_valid_; }

const std::vector<std::string>& NNGraph::inputs_tensor_meta_str() const {
  return inputs_tensor_meta_str_;
}

const std::vector<std::string>& NNGraph::outputs_tensor_meta_str() const {
  return outputs_tensor_meta_str_;
}

int64_t NNGraph::variable_op_size() const { return variable_op_names_.size(); }

const std::shared_ptr<vm::EagerBlobObjectList>& NNGraph::var_blobs() const {
  return variable_op_blobs_;
}

Maybe<void> NNGraph::RegisterAdditionalVarOpNamesAndTensorsToBeLoaded(
    const std::vector<std::string>& additional_var_names,
    const std::vector<std::shared_ptr<one::Tensor>>& additional_var_tensors) {
  CHECK_EQ_OR_RETURN(additional_var_names.size(), additional_var_tensors.size())
      << Error::RuntimeError()
      << "Number of additional variable names and tensors mismatch. "
         "Size of variable names: "
      << additional_var_names.size() << ", size of tensors: " << additional_var_tensors.size();
  CHECK_OR_RETURN(additional_variable_op_tobe_loaded_name2tensor_.empty())
      << Error::RuntimeError()
      << "The additional variables (states in Optimizer or LRScheduler) of nn.Graph " << name_
      << " are registered repeatedly.";
  FOR_RANGE(size_t, i, 0, additional_var_names.size()) {
    CHECK_OR_RETURN(additional_variable_op_tobe_loaded_name2tensor_
                        .emplace(JUST(VectorAt(additional_var_names, i)),
                                 JUST(VectorAt(additional_var_tensors, i)))
                        .second)
        << Error::RuntimeError() << "Duplicate variable name: " << additional_var_names[i];
  }
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::RegisterInputOpNamesAndTensors(
    const std::vector<std::string>& inputs_op_names,
    const std::vector<std::shared_ptr<one::Tensor>>& input_tensors) {
  CHECK_EQ_OR_RETURN(inputs_op_names.size(), input_tensors.size())
      << Error::RuntimeError()
      << "Number of input op names and tensors mismatch. "
         "Size of op names: "
      << inputs_op_names.size() << ", size of tensors: " << input_tensors.size();
  CHECK_OR_RETURN(inputs_op_names_.empty())
      << Error::RuntimeError() << "The input tensors of nn.Graph " << name_
      << " are registered repeatedly.";
  CHECK_OR_RETURN(input_tensors_valid_.empty())
      << Error::RuntimeError() << "The input tensors of nn.Graph " << name_
      << " are registered repeatedly.";
  CHECK_OR_RETURN(inputs_tensor_meta_str_.empty())
      << Error::RuntimeError() << "The input tensors of nn.Graph " << name_
      << " are registered repeatedly.";
  inputs_op_names_.assign(inputs_op_names.begin(), inputs_op_names.end());
  input_tensors_valid_.reserve(input_tensors.size());
  inputs_tensor_meta_str_.reserve(input_tensors.size());
  for (const auto& input_tensor : input_tensors) {
    input_tensors_valid_.emplace_back(JUST(GetTensorValidInCurRank(input_tensor)));
    inputs_tensor_meta_str_.emplace_back(*JUST(GetTensorMetaString(input_tensor)));
  }
  CHECK_EQ_OR_RETURN(input_tensors_valid_.size(), input_tensors.size());  // NOLINE
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::RegisterOutputOpNamesAndTensors(
    const std::vector<std::string>& outputs_op_names,
    const std::vector<std::shared_ptr<one::Tensor>>& output_tensors) {
  CHECK_EQ_OR_RETURN(outputs_op_names.size(), output_tensors.size())
      << "Number of output op names and tensors mismatch "
         "Size of op names: "
      << outputs_op_names.size() << ", size of tensors: " << output_tensors.size();
  CHECK_OR_RETURN(outputs_op_names_.empty())
      << Error::RuntimeError() << "The output tensors of nn.Graph " << name_
      << " are registered repeatedly.";
  CHECK_OR_RETURN(output_tensors_valid_.empty())
      << Error::RuntimeError() << "The output tensors of nn.Graph " << name_
      << " are registered repeatedly.";
  CHECK_OR_RETURN(outputs_tensor_meta_str_.empty())
      << Error::RuntimeError() << "The output tensors of nn.Graph " << name_
      << " are registered repeatedly.";
  outputs_op_names_.assign(outputs_op_names.begin(), outputs_op_names.end());
  output_tensors_valid_.reserve(output_tensors.size());
  outputs_tensor_meta_str_.reserve(output_tensors.size());
  for (const auto& output_tensor : output_tensors) {
    output_tensors_valid_.emplace_back(JUST(GetTensorValidInCurRank(output_tensor)));
    outputs_tensor_meta_str_.emplace_back(*JUST(GetTensorMetaString(output_tensor)));
  }
  CHECK_EQ_OR_RETURN(output_tensors_valid_.size(), output_tensors.size());  // NOLINT
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::RegisterVariableOpNamesAndTensors(
    const std::vector<std::string>& variable_op_names,
    const std::vector<std::shared_ptr<one::Tensor>>& variable_tensors) {
  JUST(vm::CurrentRankSync());
  CHECK_EQ_OR_RETURN(variable_op_names.size(), variable_tensors.size())
      << "Number of variable names and tensors mismatch. "
         "Size of variable names: "
      << variable_op_names.size() << ", size of tensors: " << variable_tensors.size();
  CHECK_ISNULL_OR_RETURN(variable_op_blobs_);
  variable_op_blobs_ = std::make_shared<vm::EagerBlobObjectList>();
  JUST(MakeEagerBlobObjectList(variable_op_blobs_.get(), variable_tensors));
  for (int32_t i = 0; i < variable_op_names.size(); ++i) {
    const std::shared_ptr<one::Tensor>& var = variable_tensors[i];
    CHECK_OR_RETURN(var->is_eager())
        << Error::InvalidValueError() << "Tensor variable to register in nn.Graph should be eager";
    const std::string& var_name = variable_op_names.at(i);
    CHECK_OR_RETURN(!var_name.empty()) << Error::InvalidValueError() << "Empty variable name";
    CHECK_OR_RETURN(variable_op_name2tensor_.emplace(var_name, var).second)
        << Error::RuntimeError() << "Duplicate variable name: " << var_name;
    CHECK_OR_RETURN(variable_op_names_.insert(var_name).second)
        << Error::RuntimeError() << "Duplicate variable name: " << var_name;
  }
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::RegisterFreeEagerTensorsToVariableOpNames() {
  JUST(vm::CurrentRankSync());
  const auto& free_eager_tensors = session_ctx_->GetFreeEagerTensorNamePairByGraphName(name_);
  for (const auto& pair : free_eager_tensors) {
    const std::string& var_name = pair.first;
    const std::shared_ptr<one::Tensor>& var = pair.second;
    CHECK_OR_RETURN(var->is_eager())
        << Error::RuntimeError() << "Free tensor variable to register in nn.Graph should be eager";
    CHECK_OR_RETURN(!var_name.empty()) << Error::RuntimeError() << "Empty variable name";
    CHECK_OR_RETURN(variable_op_name2tensor_.emplace(var_name, var).second)
        << Error::RuntimeError() << "Duplicate variable name: " << var_name;
    CHECK_OR_RETURN(additional_variable_op_name_.insert(var_name).second)
        << Error::RuntimeError() << "Duplicate variable name: " << var_name;
    CHECK_OR_RETURN(variable_op_names_.insert(var_name).second)
        << Error::RuntimeError() << "Duplicate variable name: " << var_name;
  }
  return Maybe<void>::Ok();
}

Maybe<std::vector<std::string>> NNGraph::GetAdditionalVarOpNames() const {
  std::vector<std::string> names;
  for (const auto& iter : additional_variable_op_name_) { names.push_back(iter); }
  return names;
}

Maybe<std::vector<std::shared_ptr<one::Tensor>>> NNGraph::GetAdditionalVarOpTensors() const {
  std::vector<std::shared_ptr<one::Tensor>> tensors;
  for (const auto& iter : additional_variable_op_name_) {
    auto find_iter = variable_op_name2tensor_.find(iter);
    CHECK_OR_RETURN(find_iter != variable_op_name2tensor_.end())
        << Error::RuntimeError() << "Additional variable op name " << iter << " not found.";
    tensors.push_back(find_iter->second);
  }
  return tensors;
}

Maybe<void> NNGraph::RegisterNewVariableOpInJobPass() {
  OpGraph op_graph(job_);
  JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
    if (op_node->op().op_conf().has_variable_conf() == false) { return Maybe<void>::Ok(); }
    const Operator& variable_op = op_node->op();
    const VariableOpConf& var_conf = variable_op.op_conf().variable_conf();
    const std::string& var_name = variable_op.op_name();
    CHECK_OR_RETURN(var_conf.has_initializer())
        << Error::RuntimeError() << "nn.Graph ONLY support variable op with initializer conf.";
    if (var_conf.initializer().has_constant_conf()
        || var_conf.initializer().has_constant_int_conf() /* vairable ops inserted by system */) {
      CHECK_OR_RETURN(variable_op_names_.insert(var_name).second)
          << Error::RuntimeError() << "Variable_op_name: " << var_name
          << " has been added in nn.Graph: " << name_;
      CHECK_OR_RETURN(
          variable_op_name2tensor_.insert({var_name, std::shared_ptr<one::Tensor>()}).second)
          << Error::RuntimeError() << "Variable Tensor with op_name: " << var_name
          << " has been add in nn.Graph: " << name_;
      CHECK_OR_RETURN(additional_variable_op_name_.insert(var_name).second)
          << Error::RuntimeError() << "Variable Tensor with op_name: " << var_name
          << " has been add in nn.Graph: " << name_;
    } else /* vairable ops from user code */ {
      CHECK_OR_RETURN(var_conf.initializer().has_empty_conf())
          << Error::RuntimeError() << "nn.Graph ONLY support variable_op with empty conf, "
          << "because variable is inited by eager tensor. "
          << "This error variable conf is: " << variable_op.op_conf().DebugString()
          << " in nn.Graph " << name_;
      CHECK_OR_RETURN(variable_op_names_.find(var_name) != variable_op_names_.end())
          << Error::RuntimeError() << var_name
          << " must be a variable created in nn.Graph: " << name_;
    }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::DeleteOutdatedVariableInVariableTensorMgr() {
  const auto& var_get_func = [&]() -> Maybe<std::set<std::string>> {
    std::set<std::string> variable_names_;
    OpGraph op_graph(job_);
    JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
      if (op_node->op().op_conf().has_variable_conf() == false) { return Maybe<void>::Ok(); }
      variable_names_.insert(op_node->op().op_name());
      return Maybe<void>::Ok();
    }));
    return variable_names_;
  };
  std::set<std::string> variable_names = *JUST(var_get_func());

  auto mgr = Singleton<VariableTensorMgr>::Get();
  for (auto& name : mgr->DumpNames()) {
    if (variable_names.find(name) == variable_names.end()) { mgr->Delete(name); }
  }
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::AlignStatesAfterLogicalGraphCompile() {
  auto compile_tc = std::make_unique<CostCounter<std::chrono::seconds>>(true, true);
  JUST(RegisterFreeEagerTensorsToVariableOpNames());
  JUST(RegisterNewVariableOpInJobPass());
  JUST(DeleteOutdatedVariableInVariableTensorMgr());
  // NOTE(chengcheng): TensorNameScope need to be cleared after current graph is built.
  one::TensorNameScope::Global()->Clear();
  // Clear all backward pass scope
  ClearAllBackwardPassScope();
  compile_tc->Count("[GraphCompile]" + name_ + " AlignStates", 0);
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::CompleteLogicalGraphForRuntime() {
  auto compile_tc = std::make_unique<CostCounter<std::chrono::seconds>>(true, true);
  // A global variable to get graph configurations.
  auto current_graph_config = std::make_unique<GlobalJobDescScope>(job_.job_conf(), job_id());
  // NOTE(chengcheng): do job compeleter for each rank.
  JUST(JobCompleter::Complete(&job_));
  compile_tc->Count("[GraphCompile]" + name_ + " CompleteJob", 0);
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::BuildWithNewInputFromSharedGraph(
    const std::vector<std::string>& shared_inputs_op_names,
    const std::vector<std::shared_ptr<one::Tensor>>& new_input_tensors,
    const std::vector<std::string>& shared_op_names_from_ordered_original_graph,
    const std::string& new_serialized_original_job) {
  CHECK_EQ_OR_RETURN(shared_inputs_op_names.size(), new_input_tensors.size());  // NOLINE
  auto compile_tc = std::make_unique<CostCounter<std::chrono::seconds>>(true, true);
  // Register inputs.
  JUST(RegisterInputOpNamesAndTensors(shared_inputs_op_names, new_input_tensors));

  // Generate new input tensor getter.
  HashMap<std::string, std::shared_ptr<one::Tensor>> input_name2tensor;
  for (int64_t idx = 0; idx < shared_inputs_op_names.size(); ++idx) {
    input_name2tensor.emplace(shared_inputs_op_names[idx], new_input_tensors[idx]);
  }
  const auto& InputTensor4Name =
      [&input_name2tensor](const std::string& op_name) -> Maybe<std::shared_ptr<one::Tensor>> {
    auto iter = input_name2tensor.find(op_name);
    CHECK_OR_RETURN(iter != input_name2tensor.end())
        << "Can't find input tensor of " << op_name << ".";
    return iter->second;
  };

  // Generate new OperatorConf getter.
  Job new_build_original_job;
  CHECK_OR_RETURN(new_build_original_job.ParseFromString(new_serialized_original_job))
      << "nn.Graph " << name_ << " parse job proto of new build graph failed.";
  CHECK_EQ_OR_RETURN(new_build_original_job.net().op_size(),
                     shared_op_names_from_ordered_original_graph.size())
      << "nn.Graph " << name_
      << " new_build_original_job op size and shared_op_names_from_ordered_original_graph size are "
         "not "
         "equal.";
  HashMap<std::string, const OperatorConf*> shared_op_name2_new_op;
  for (int64_t op_idx = 0; op_idx < shared_op_names_from_ordered_original_graph.size(); ++op_idx) {
    // Assume that the new graph and the shared graph from nn.Graph.build have the same op order.
    const auto& op = new_build_original_job.mutable_net()->mutable_op()->at(op_idx);
    shared_op_name2_new_op.emplace(shared_op_names_from_ordered_original_graph[op_idx], &op);
  }
  const auto& NewOp4SharedOpName =
      [&shared_op_name2_new_op](const std::string& shared_op_name) -> Maybe<const OperatorConf*> {
    auto iter = shared_op_name2_new_op.find(shared_op_name);
    CHECK_OR_RETURN(iter != shared_op_name2_new_op.end())
        << "Can't find new operator conf of " << shared_op_name << ".";
    return iter->second;
  };

  // A global variable to get graph configurations.
  auto current_graph_config = std::make_unique<GlobalJobDescScope>(job_.job_conf(), job_id());
  // NOTE(chengcheng): do job compeleter for each rank.
  JUST(JobCompleter::UpdateSharedGraphForNewInput(&job_, InputTensor4Name, NewOp4SharedOpName));
  compile_tc->Count("[GraphCompile]" + name_ + " CompleteJob", 0);
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::CompilePlanForRuntime() {
  auto compile_tc = std::make_unique<CostCounter<std::chrono::seconds>>(true, true);
  // A global variable to get graph configurations.
  auto current_graph_config = std::make_unique<GlobalJobDescScope>(job_.job_conf(), job_id());
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    // TODO(chengcheng): new memory reused by chunk
    Compiler().Compile(&job_, &plan_);
    auto sub_compile_tc = std::make_unique<CostCounter<std::chrono::seconds>>(true, true);
    PlanUtil::GenMemBlockAndChunkWithVariableOpNames4Plan(&plan_, variable_op_names_);
    sub_compile_tc->Count("[GraphCompile]" + name_ + " GenMemBlockAndChunk", 1, true);
    if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
      TeePersistentLogStream::Create("job_" + name_ + "_plan")->Write(plan_);
      PlanUtil::ToDotFile(plan_, "job_" + name_ + "_plan.dot");
    }
    sub_compile_tc->Count("[GraphCompile]" + name_ + " LogPlan", 1, true);
    PlanUtil::GenRegisterHint(&plan_);
    sub_compile_tc->Count("[GraphCompile]" + name_ + " GenRegisterHint", 1, true);
    // TODO(chengcheng): test collective boxing for multi-job.
    PlanUtil::GenCollectiveBoxingPlan(&job_, &plan_);
    sub_compile_tc->Count("[GraphCompile]" + name_ + " GenCollectiveBoxingPlan", 1, true);
    PlanUtil::DumpCtrlRegstInfoToPlan(&plan_);
    sub_compile_tc->Count("[GraphCompile]" + name_ + " DumpCtrlRegstInfoToPlan", 1, true);
    PlanUtil::PlanMemoryLog(&plan_, name_);
    if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
      PlanUtil::GenLightPlan(&plan_, name_);
    }
    sub_compile_tc->Count("[GraphCompile]" + name_ + " GenMemAndLightPlanLog", 1, true);
  }
  compile_tc->Count("[GraphCompile]" + name_ + " CompilePlan", 0);
  if (GlobalProcessCtx::WorldSize() > 1) {
    std::string plan_name = "plan:" + job_name();
    if (GlobalProcessCtx::IsThisProcessMaster()) {
      // TODO(chengcheng): split plan for each rank.
      Singleton<CtrlClient>::Get()->PushKV(plan_name, plan_);
    } else {
      Singleton<CtrlClient>::Get()->PullKV(plan_name, &plan_);
    }
    OF_SESSION_BARRIER();
    // NOTE(zwx): After barrier plan is synchronized between all ranks,
    //     then it can be cleared for saving mem.
    if (GlobalProcessCtx::IsThisProcessMaster()) {
      Singleton<CtrlClient>::Get()->ClearKV(plan_name);
    }
  }
  compile_tc->Count("[GraphCompile]" + name_ + " SyncPlan", 0, true);
  // NOTE(chengcheng): recovery op_attr
  PlanUtil::PopulateOpAttribute(&plan_, plan_.job_id2op_attribute_ref_table());
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::InitRuntime() {
  CHECK_OR_RETURN(!runtime_inited_)
      << Error::RuntimeError() << "nn.Graph runtime is already initialized";

  auto compile_tc = std::make_unique<CostCounter<std::chrono::seconds>>(true, true);
  NewRuntimeBuffers();

  JUST(GetVariableRealBlobAfterSyncPlan());

  // NOTE(strint): Do memory shrink to free cached memory in eager VM before graph runtime init.
  JUST(vm::CurrentRankSync());
  auto* vm = JUST(SingletonMaybe<VirtualMachine>());
  JUST(vm->ShrinkAllMem());

  runtime_.reset(new Runtime(plan_, variable_op_name2eager_blob_object_));
  compile_tc->Count("[GraphCompile]" + name_ + " InitRuntime", 0, true);
  JUST(LogProgress("[GraphCompile]" + name_ + " Done", true));

  runtime_inited_ = true;
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::CompileAndInitRuntime() {
  JUST(AlignStatesAfterLogicalGraphCompile());
  JUST(CompleteLogicalGraphForRuntime());
  JUST(CompilePlanForRuntime());
  JUST(InitRuntime());
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::GetVariableRealBlobAfterSyncPlan() {
  CHECK_OR_RETURN(variable_op_name2eager_blob_object_.empty())
      << Error::RuntimeError() << kOfBugIssueUploadPrompt;
  JUST(vm::CurrentRankSync());
  // Create or Rebuild variable, then get the real blob.
  for (const std::string& var_name : variable_op_names_) {
    auto iter = variable_op_name2tensor_.find(var_name);
    CHECK_OR_RETURN(iter != variable_op_name2tensor_.end())
        << Error::RuntimeError() << "variable op name " << var_name << " not found.";
    std::shared_ptr<one::Tensor> tensor = iter->second;
    vm::EagerBlobObject* var_blob = nullptr;
    if (plan_.job_id2op_attribute_ref_table().at(job_id_).op_name2op_attribute().find(var_name)
        == plan_.job_id2op_attribute_ref_table().at(job_id_).op_name2op_attribute().end()) {
      // Deal with variable tensor not used in nn.Graph build.
      CHECK_OR_RETURN(tensor != NULL)
          << Error::RuntimeError() << "The tensor of " << var_name
          << " does not exist in the job, so it's not created in nn.Graph and cannot be NULL.";
      if (tensor->is_global()) {
        const std::shared_ptr<one::LocalTensor> local_var = JUST(tensor->cur_rank_phy_tensor());
        var_blob = JUST(local_var->eager_blob_object()).get();
      } else {
        var_blob = JUST(tensor->eager_blob_object()).get();
      }
    } else if (/*is_null=*/!tensor) {
      // Deal with tensors which are not in the nn.Module.
      // We can call these tensors as additional variables.
      const auto& op_attribute =
          plan_.job_id2op_attribute_ref_table().at(job_id_).op_name2op_attribute().at(var_name);
      // NOTE(chengcheng): handle constant variable created by job pass
      Symbol<ParallelDesc> placement(op_attribute.parallel_conf_signature().op_parallel_conf());
      NdSbp nd_sbp(NdSbpSignature(op_attribute.nd_sbp_signature()).bn_in_op2nd_sbp().at("out"));
      const BlobDesc blob_desc(
          op_attribute.logical_blob_desc_signature().bn_in_op2blob_desc().at("out"));
      DType dtype(blob_desc.data_type());
      std::shared_ptr<std::vector<Symbol<SbpParallel>>> sbp_tuple =
          JUST(GetSbpList(Symbol<NdSbp>(nd_sbp)));

      auto load_tensor_iter = additional_variable_op_tobe_loaded_name2tensor_.find(var_name);
      if (load_tensor_iter == additional_variable_op_tobe_loaded_name2tensor_.end()) {
        // Create a additional variable tensor
        Scalar value;
        const VariableOpConf& var_conf = op_attribute.op_conf().variable_conf();
        if (var_conf.initializer().has_constant_conf()) {
          value = var_conf.initializer().constant_conf().value();
        } else if (var_conf.initializer().has_constant_int_conf()) {
          value = var_conf.initializer().constant_int_conf().value();
        } else {
          OF_UNIMPLEMENTED();
        }
        // NOTE(chengcheng): New EagerTensor need set LazyMode false.
        auto lazy_mode_disabled_guard = LazyMode::Guard(/*is_enabled*/ false);
        tensor = JUST(one::functional::GlobalConstant(blob_desc.shape(), value,
                                                      Symbol<DType>(dtype), placement, *sbp_tuple));
        JUST(vm::CurrentRankSync());
        VLOG(2) << "Lazy nn.Graph name " << name_ << " op: " << op_attribute.op_conf().name()
                << " created in JobPass, nn.Graph has created a eager tensor for this variable.\n";
      } else {
        // Load a additional variable tensor
        auto lazy_mode_disabled_guard = LazyMode::Guard(/*is_enabled*/ false);
        std::vector<Symbol<SbpParallel>> grad_sbp_tuple;
        // To consistent from a local or global tensor.
        bool check_meta = load_tensor_iter->second->is_global() ? false : true;
        tensor = JUST(one::functional::ToGlobal(load_tensor_iter->second, placement, *sbp_tuple,
                                                grad_sbp_tuple, check_meta, /*copy=*/false));
        JUST(vm::CurrentRankSync());
        VLOG(2) << "Lazy nn.Graph name " << name_ << " op: " << op_attribute.op_conf().name()
                << " created in JobPass, nn.Graph has loaded the tensor from state dict for this "
                   "variable.\n";
      }
      // Register
      JUST(MapAt(variable_op_name2tensor_, var_name)) = tensor;
      // NOTE(chengcheng): Just for tensor lifetime hold by session context in graph lifetime
      // valid.
      session_ctx_->StoreFreeEagerTensorWithNameByGraphName(name_, tensor, var_name);

      const std::shared_ptr<one::LocalTensor> local_var = JUST(tensor->cur_rank_phy_tensor());
      var_blob = JUST(local_var->eager_blob_object()).get();
    } else if (tensor->is_global()) {
      // Deal with tensors which need to change sbp.
      NdSbpSignature var_nd_sbp_signature = NdSbpSignature(plan_.job_id2op_attribute_ref_table()
                                                               .at(job_id_)
                                                               .op_name2op_attribute()
                                                               .at(var_name)
                                                               .nd_sbp_signature());
      NdSbp optimized_nd_sbp = var_nd_sbp_signature.bn_in_op2nd_sbp().at("out");
      // Change variable tensor's impl with new sbp when job pass has changed their sbp.
      if (*JUST(tensor->nd_sbp()) != optimized_nd_sbp) {
        VLOG(2) << "Graph with name " << name_ << " variable with name `" << var_name
                << "` changes its' sbp from " << NdSbpToString(*JUST(tensor->nd_sbp())) << " to "
                << NdSbpToString(optimized_nd_sbp) << " after compile optimization.";
        std::vector<Symbol<SbpParallel>> optimized_sbp_parallels;
        for (int i = 0; i < optimized_nd_sbp.sbp_parallel_size(); ++i) {
          optimized_sbp_parallels.emplace_back(optimized_nd_sbp.sbp_parallel(i));
        }
        {
          auto lazy_mode_disabled_guard = LazyMode::Guard(/* is_enabled */ false);
          const auto& new_tensor = JUST(one::functional::ToGlobal(
              tensor, JUST(tensor->parallel_desc()), optimized_sbp_parallels, {},
              /* check_meta */ false, /*copy=*/false));
          JUST(vm::CurrentRankSync());
          // Use tensor.set_data inferface and make new TensorImpl instead of the old one.
          JUST(tensor->set_data(new_tensor));
        }
      }
      const std::shared_ptr<one::LocalTensor> local_var = JUST(tensor->cur_rank_phy_tensor());
      var_blob = JUST(local_var->eager_blob_object()).get();
    } else {
      var_blob = JUST(tensor->eager_blob_object()).get();
    }
    CHECK_OR_RETURN(var_blob != nullptr) << Error::RuntimeError() << kOfBugIssueUploadPrompt;
    CHECK_OR_RETURN(variable_op_name2eager_blob_object_.emplace(var_name, var_blob).second)
        << Error::RuntimeError() << kOfBugIssueUploadPrompt;
  }
  // Initialize or check mem_ptr_for_allocation_computation_pipelining by TouchTensors instruction.
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    auto eager_blob_objects = std::make_shared<vm::EagerBlobObjectList>();
    for (const auto& pair : variable_op_name2eager_blob_object_) {
      eager_blob_objects->push_back(pair.second->shared_from_this());
    }
    return builder->TouchTensors(eager_blob_objects);
  }));
  JUST(vm::CurrentRankSync());
  // Clear after load additional variable is finished.
  additional_variable_op_tobe_loaded_name2tensor_.clear();
  return Maybe<void>::Ok();
}

void NNGraph::NewRuntimeBuffers() {
  // NOTE(chengcheng):
  //   1. The BufferSize comes from job_conf.concurrency_width configured by user (default = 128)
  //   2. In Pipeline Parallelism, this value need greater than pipeline stage num for pipelining.
  size_t concurrency_width = job_.job_conf().concurrency_width();
  {
    auto* buffer_mgr = Singleton<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
    buffer_mgr->NewBuffer(GetSourceTickBufferName(name_), concurrency_width);
    buffer_mgr->NewBuffer(GetCallbackNotifierBufferName(name_), concurrency_width);
  }
  {
    auto* buffer_mgr = Singleton<BufferMgr<std::shared_ptr<CriticalSectionInstance>>>::Get();
    buffer_mgr->NewBuffer(GetInputCriticalSectionWaitBufferName(name_), concurrency_width);
    buffer_mgr->NewBuffer(GetInputCriticalSectionCallbackBufferName(name_), concurrency_width);
    buffer_mgr->NewBuffer(GetOutputCriticalSectionWaitBufferName(name_), concurrency_width);
    buffer_mgr->NewBuffer(GetOutputCriticalSectionCallbackBufferName(name_), concurrency_width);
    for (const std::string& input_op_name : inputs_op_names_) {
      buffer_mgr->NewBuffer(GetInputBufferName(name_, input_op_name), concurrency_width);
    }
    for (const std::string& output_op_name : outputs_op_names_) {
      buffer_mgr->NewBuffer(GetOutputBufferName(name_, output_op_name), concurrency_width);
    }
  }
}

void NNGraph::CloseRuntimeBuffers() {
  if (runtime_inited_) {
    {
      auto* buffer_mgr = Singleton<BufferMgr<std::shared_ptr<CriticalSectionInstance>>>::Get();
      for (const std::string& output_op_name : outputs_op_names_) {
        buffer_mgr->Get(GetOutputBufferName(name_, output_op_name))->Close();
      }
      for (const std::string& input_op_name : inputs_op_names_) {
        buffer_mgr->Get(GetInputBufferName(name_, input_op_name))->Close();
      }
      buffer_mgr->Get(GetOutputCriticalSectionCallbackBufferName(name_))->Close();
      buffer_mgr->Get(GetOutputCriticalSectionWaitBufferName(name_))->Close();
      buffer_mgr->Get(GetInputCriticalSectionCallbackBufferName(name_))->Close();
      buffer_mgr->Get(GetInputCriticalSectionWaitBufferName(name_))->Close();
    }
    {
      auto* buffer_mgr = Singleton<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
      buffer_mgr->Get(GetCallbackNotifierBufferName(name_))->Close();
      buffer_mgr->Get(GetSourceTickBufferName(name_))->Close();
    }
  }
}

Maybe<void> RunLazyNNGraph(const one::TensorTuple& inputs, const one::TensorTuple& outputs,
                           const std::shared_ptr<NNGraph>& nn_graph) {
  CHECK_EQ_OR_RETURN(inputs.size(), nn_graph->inputs_op_names().size())
      << Error::RuntimeError()
      << "Number of inputs and NNGraph::inputs_op_names mismatch. "
         "Size of inputs: "
      << inputs.size()
      << ", size of NNGraph::inputs_op_names: " << nn_graph->inputs_op_names().size();
  CHECK_EQ_OR_RETURN(outputs.size(), nn_graph->outputs_op_names().size())
      << Error::RuntimeError()
      << "Number of outputs and NNGraph::outputs_op_names mismatch. "
         "Size of outputs: "
      << outputs.size()
      << ", size of NNGraph::outputs_op_names: " << nn_graph->outputs_op_names().size();
  // NOTE(chengcheng):
  //   parameters not used in LaunchLazyJobInstrucntion;
  //   the args: parameters is all variable tensor hold by nn.Graph
  //   but the NNGraph::variable_op_size may has FreeEagerTensor as sepcial variable op.
  CHECK_LE_OR_RETURN(nn_graph->var_blobs()->size(), nn_graph->variable_op_size())
      << Error::RuntimeError() << "Parameter size should be less than or equal to variable size";
  for (int i = 0; i < inputs.size(); ++i) {
    // TODO(chengcheng, liufengwei):
    //   use TensorMeta.to_string and equal.
    std::string tensor_meta_str = *JUST(GetTensorMetaString(inputs.at(i)));
    const std::string& static_meta_str = nn_graph->inputs_tensor_meta_str().at(i);
    CHECK_OR_RETURN(static_meta_str == tensor_meta_str)
        << Error::RuntimeError()
        << "nn.Graph ONLY accepts static inputs tensor meta, please check whether your input "
        << "tensor meta each step is the same as the input of first call graph.\nThe excepted "
        << "tensor meta is: " << static_meta_str
        << ", but the actual tensor meta is: " << tensor_meta_str << ". The input index is " << i
        << ".";
  }
  for (int i = 0; i < outputs.size(); ++i) {
    CHECK_OR_RETURN(nn_graph->outputs_tensor_meta_str().at(i)
                    == *JUST(GetTensorMetaString(outputs.at(i))))
        << Error::RuntimeError() << "Output tensor meta string mismatch";
  }
  vm::EagerBlobObjectList input_blobs;
  vm::EagerBlobObjectList output_blobs;
  JUST(MakeEagerBlobObjectList(&input_blobs, inputs));
  JUST(MakeEagerBlobObjectList(&output_blobs, outputs));
  const auto& input_blob_list_ptr =
      std::make_shared<const vm::EagerBlobObjectList>(std::move(input_blobs));
  const auto& output_blob_list_ptr =
      std::make_shared<const vm::EagerBlobObjectList>(std::move(output_blobs));
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->LaunchLazyJob(input_blob_list_ptr, output_blob_list_ptr, nn_graph->var_blobs(),
                                  nn_graph);
  }));
  return Maybe<void>::Ok();
}

Maybe<void> SoftSyncNNGraphBuffers(const one::TensorTuple& buffers,
                                   const std::shared_ptr<NNGraph>& nn_graph) {
  const auto& eager_blob_objects = std::make_shared<vm::EagerBlobObjectList>();
  JUST(MakeEagerBlobObjectList(eager_blob_objects.get(), buffers));
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->SoftSyncNNGraphBuffers(eager_blob_objects, nn_graph);
  }));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
