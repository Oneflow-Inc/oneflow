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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/nd_sbp.h"
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

Maybe<void> NNGraph::RegisterAdditionalVarOpNamesAndTensorsToBeLoaded(
    const std::vector<std::string>& additional_var_names,
    const std::vector<std::shared_ptr<one::Tensor>>& additional_var_tensors) {
  CHECK_EQ_OR_RETURN(additional_var_names.size(), additional_var_tensors.size());
  CHECK_OR_RETURN(additional_variable_op_tobe_loaded_name2tensor_.empty())
      << " The additional variables (states in Optimizer or LRScheduler) of nn.Graph " << name_
      << " are register repeatedly.";
  FOR_RANGE(size_t, i, 0, additional_var_names.size()) {
    CHECK_OR_RETURN(additional_variable_op_tobe_loaded_name2tensor_
                        .emplace(JUST(VectorAt(additional_var_names, i)),
                                 JUST(VectorAt(additional_var_tensors, i)))
                        .second);
  }
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::RegisterInputOpNamesAndTensors(
    const std::vector<std::string>& inputs_op_names,
    const std::vector<std::shared_ptr<one::Tensor>>& input_tensors) {
  CHECK_EQ_OR_RETURN(inputs_op_names.size(), input_tensors.size());
  CHECK_OR_RETURN(inputs_op_names_.empty())
      << " The input tensors of nn.Graph " << name_ << " are register repeatedly.";
  CHECK_OR_RETURN(input_tensors_valid_.empty());
  CHECK_OR_RETURN(inputs_tensor_meta_str_.empty());
  inputs_op_names_.assign(inputs_op_names.begin(), inputs_op_names.end());
  input_tensors_valid_.reserve(input_tensors.size());
  inputs_tensor_meta_str_.reserve(input_tensors.size());
  for (const auto& input_tensor : input_tensors) {
    input_tensors_valid_.emplace_back(JUST(GetTensorValidInCurRank(input_tensor)));
    inputs_tensor_meta_str_.emplace_back(*JUST(GetTensorMetaString(input_tensor)));
  }
  CHECK_EQ_OR_RETURN(input_tensors_valid_.size(), input_tensors.size());
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::RegisterOutputOpNamesAndTensors(
    const std::vector<std::string>& outputs_op_names,
    const std::vector<std::shared_ptr<one::Tensor>>& output_tensors) {
  CHECK_EQ_OR_RETURN(outputs_op_names.size(), output_tensors.size());
  CHECK_OR_RETURN(outputs_op_names_.empty())
      << " The output tensors of nn.Graph " << name_ << " are register repeatedly.";
  CHECK_OR_RETURN(output_tensors_valid_.empty());
  CHECK_OR_RETURN(outputs_tensor_meta_str_.empty());
  outputs_op_names_.assign(outputs_op_names.begin(), outputs_op_names.end());
  output_tensors_valid_.reserve(output_tensors.size());
  outputs_tensor_meta_str_.reserve(output_tensors.size());
  for (const auto& output_tensor : output_tensors) {
    output_tensors_valid_.emplace_back(JUST(GetTensorValidInCurRank(output_tensor)));
    outputs_tensor_meta_str_.emplace_back(*JUST(GetTensorMetaString(output_tensor)));
  }
  CHECK_EQ_OR_RETURN(output_tensors_valid_.size(), output_tensors.size());
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::RegisterVariableOpNamesAndTensors(
    const std::vector<std::string>& variable_op_names,
    const std::vector<std::shared_ptr<one::Tensor>>& variable_tensors) {
  JUST(vm::CurrentRankSync());
  CHECK_EQ_OR_RETURN(variable_op_names.size(), variable_tensors.size());
  for (int32_t i = 0; i < variable_op_names.size(); ++i) {
    const std::shared_ptr<one::Tensor>& var = variable_tensors.at(i);
    CHECK_OR_RETURN(var->is_eager());
    const std::string& var_name = variable_op_names.at(i);
    CHECK_OR_RETURN(!var_name.empty());
    CHECK_OR_RETURN(variable_op_name2tensor_.emplace(var_name, var).second);
    CHECK_OR_RETURN(variable_op_names_.insert(var_name).second);
  }
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::RegisterFreeEagerTensorsToVariableOpNames() {
  JUST(vm::CurrentRankSync());
  const auto& free_eager_tensors = session_ctx_->GetFreeEagerTensorNamePairByGraphName(name_);
  for (const auto& pair : free_eager_tensors) {
    const std::string& var_name = pair.first;
    const std::shared_ptr<one::Tensor>& var = pair.second;
    CHECK_OR_RETURN(var->is_eager());
    CHECK_OR_RETURN(!var_name.empty());
    CHECK_OR_RETURN(variable_op_name2tensor_.emplace(var_name, var).second);
    CHECK_OR_RETURN(additional_variable_op_name_.insert(var_name).second);
    CHECK_OR_RETURN(variable_op_names_.insert(var_name).second);
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
    CHECK_OR_RETURN(find_iter != variable_op_name2tensor_.end());
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
        << " nn.Graph ONLY support variable op with initializer conf.";
    if (var_conf.initializer().has_constant_conf()
        || var_conf.initializer().has_constant_int_conf() /* vairable ops inserted by system */) {
      CHECK_OR_RETURN(variable_op_names_.insert(var_name).second)
          << " ERROR! variable_op_name: " << var_name << " has been add in nn.Graph: " << name_;
      CHECK_OR_RETURN(
          variable_op_name2tensor_.insert({var_name, std::shared_ptr<one::Tensor>()}).second)
          << " ERROR! variable Tensor with op_name: " << var_name
          << " has been add in nn.Graph: " << name_;
      CHECK_OR_RETURN(additional_variable_op_name_.insert(var_name).second)
          << " ERROR! variable Tensor with op_name: " << var_name
          << " has been add in nn.Graph: " << name_;
    } else /* vairable ops from user code */ {
      CHECK_OR_RETURN(var_conf.initializer().has_empty_conf())
          << " nn.Graph ONLY support variable_op with empty conf,"
          << " because variable is inited by eager tensor."
          << " This error variable conf is : " << variable_op.op_conf().DebugString()
          << " in nn.Graph " << name_;
      CHECK_OR_RETURN(variable_op_names_.find(var_name) != variable_op_names_.end())
          << " ERROR! " << var_name << " must be a variable created in nn.Graph: " << name_;
    }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}
Maybe<void> NNGraph::DeleteOutdatedVariableInVariableTensorMgr() {
  std::set<std::string> variable_names = [&]() -> Maybe<std::set<std::string>> {
    std::set<std::string> variable_names_;
    OpGraph op_graph(job_);
    JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
      if (op_node->op().op_conf().has_variable_conf() == false) { return Maybe<void>::Ok(); }
      variable_names_.insert(op_node->op().op_name());
      return Maybe<void>::Ok();
    }));
    return variable_names_;
  }()
                                                      .GetOrThrow();

  auto mgr = Singleton<VariableTensorMgr>::Get();
  for (auto& name : mgr->DumpNames()) {
    if (variable_names.find(name) == variable_names.end()) { mgr->Delete(name); }
  }
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::CompileAndInitRuntime() {
  CHECK_OR_RETURN(!runtime_inited_);
  JUST(RegisterFreeEagerTensorsToVariableOpNames());
  JUST(RegisterNewVariableOpInJobPass());
  JUST(DeleteOutdatedVariableInVariableTensorMgr());

  // NOTE(chengcheng): TensorNameScope need to be cleared after current graph is built.
  one::TensorNameScope::Global()->Clear();

  // NOTE(chengcheng): Singleton<JobDesc> need be clear before GlobalJobDescScope construct.
  if (Singleton<JobDesc>::Get() != nullptr) { Singleton<JobDesc>::Delete(); }

  auto scope = std::make_unique<GlobalJobDescScope>(job_.job_conf(), job_id_);

  // NOTE(chengcheng): do job compeleter for each rank.
  JUST(JobCompleter().Complete(&job_));

  if (GlobalProcessCtx::IsThisProcessMaster()) {
    double start = GetCurTime();
    // TODO(chengcheng): new memory reused by chunk
    Compiler().Compile(&job_, &plan_);
    PlanUtil::GenMemBlockAndChunkWithVariableOpNames4Plan(&plan_, variable_op_names_);

    VLOG(1) << "Graph name: " << name_ << " compile time: " << (GetCurTime() - start) / 1000000000.0
            << " seconds.";
    if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
      TeePersistentLogStream::Create("job_" + name_ + "_plan")->Write(plan_);
      PlanUtil::ToDotFile(plan_, "job_" + name_ + "_plan.dot");
    }
    PlanUtil::GenRegisterHint(&plan_);
    // TODO(chengcheng): test collective boxing for multi-job.
    PlanUtil::GenCollectiveBoxingPlan(&job_, &plan_);
    // PlanUtil::SetForceInplaceMemBlock(&plan_); NOTE(chengcheng): only for ssp.
    PlanUtil::DumpCtrlRegstInfoToPlan(&plan_);
    PlanUtil::PlanMemoryLog(&plan_, name_);
    if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
      PlanUtil::GenLightPlan(&plan_, name_);
    }
  }
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
  // NOTE(chengcheng): recovery op_attr
  PlanUtil::PopulateOpAttribute(&plan_, plan_.job_id2op_attribute_ref_table());

  NewRuntimeBuffers();

  JUST(GetVariableRealBlobAfterSyncPlan());

  // NOTE(strint): Do memory shrink to free cached memory in eager VM before graph runtime init.
  JUST(vm::CurrentRankSync());
  auto* vm = JUST(SingletonMaybe<VirtualMachine>());
  JUST(vm->ShrinkAllMem());

  runtime_.reset(new Runtime(plan_, variable_op_name2eager_blob_object_));
  runtime_inited_ = true;
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::GetVariableRealBlobAfterSyncPlan() {
  CHECK_OR_RETURN(variable_op_name2eager_blob_object_.empty()) << kOfBugIssueUploadPrompt;
  JUST(vm::CurrentRankSync());
  // Create or Rebuild variable, then get the real blob.
  for (const std::string& var_name : variable_op_names_) {
    auto iter = variable_op_name2tensor_.find(var_name);
    CHECK_OR_RETURN(iter != variable_op_name2tensor_.end()) << var_name << " not found.";
    std::shared_ptr<one::Tensor> tensor = iter->second;
    vm::EagerBlobObject* var_blob = nullptr;
    if (plan_.job_id2op_attribute_ref_table().at(job_id_).op_name2op_attribute().find(var_name)
        == plan_.job_id2op_attribute_ref_table().at(job_id_).op_name2op_attribute().end()) {
      // Deal with variable tensor not used in nn.Graph build.
      CHECK(tensor != NULL)
          << "the tensor of " << var_name
          << " is not existed in job, so it's not created in nn.Graph and cannot be NULL.";
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
                                                grad_sbp_tuple, check_meta));
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
          const auto& new_tensor =
              JUST(one::functional::ToGlobal(tensor, JUST(tensor->parallel_desc()),
                                             optimized_sbp_parallels, {}, /* check_meta */ false));
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
    CHECK_OR_RETURN(var_blob != nullptr) << kOfBugIssueUploadPrompt;
    CHECK_OR_RETURN(variable_op_name2eager_blob_object_.emplace(var_name, var_blob).second)
        << kOfBugIssueUploadPrompt;
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

namespace {

Maybe<void> MakeEagerBlobObjectList(vm::EagerBlobObjectList* blob_list,
                                    const one::TensorTuple& tensor_list) {
  blob_list->reserve(tensor_list.size());
  for (const auto& tensor : tensor_list) {
    CHECK_OR_RETURN(tensor->is_eager());
    if (tensor->is_global()) {
      blob_list->emplace_back(JUST(JUST(tensor->cur_rank_phy_tensor())->eager_blob_object()));
    } else {
      blob_list->emplace_back(JUST(tensor->eager_blob_object()));
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> RunLazyNNGraph(const one::TensorTuple& inputs, const one::TensorTuple& outputs,
                           const one::TensorTuple& parameters,
                           const std::shared_ptr<NNGraph>& nn_graph) {
  CHECK_EQ_OR_RETURN(inputs.size(), nn_graph->inputs_op_names().size());
  CHECK_EQ_OR_RETURN(outputs.size(), nn_graph->outputs_op_names().size());
  // NOTE(chengcheng):
  //   parameters not used in LaunchLazyJobInstrucntion;
  //   the args: parameters is all variable tensor hold by nn.Graph
  //   but the NNGraph::variable_op_size may has FreeEagerTensor as sepcial variable op.
  CHECK_LE_OR_RETURN(parameters.size(), nn_graph->variable_op_size());
  for (int i = 0; i < inputs.size(); ++i) {
    // TODO(chengcheng, liufengwei):
    //   use TensorMeta.to_string and equal.
    std::string tensor_meta_str = *JUST(GetTensorMetaString(inputs.at(i)));
    const std::string& static_meta_str = nn_graph->inputs_tensor_meta_str().at(i);
    CHECK_OR_RETURN(static_meta_str == tensor_meta_str)
        << "\n  nn.Graph ONLY accepts static inputs tensor meta, please check whether your input "
        << "tensor meta each step is the same as the input of first call graph. \n  The excepted "
        << "tensor meta is : ( \n  " << static_meta_str
        << " \n) , but the actual tensor meta is : ( \n  " << tensor_meta_str << " \n)";
  }
  for (int i = 0; i < outputs.size(); ++i) {
    CHECK_OR_RETURN(nn_graph->outputs_tensor_meta_str().at(i)
                    == *JUST(GetTensorMetaString(outputs.at(i))));
  }
  vm::EagerBlobObjectList input_blobs;
  vm::EagerBlobObjectList output_blobs;
  vm::EagerBlobObjectList var_blobs;
  JUST(MakeEagerBlobObjectList(&input_blobs, inputs));
  JUST(MakeEagerBlobObjectList(&output_blobs, outputs));
  JUST(MakeEagerBlobObjectList(&var_blobs, parameters));
  const auto& input_blob_list_ptr =
      std::make_shared<const vm::EagerBlobObjectList>(std::move(input_blobs));
  const auto& output_blob_list_ptr =
      std::make_shared<const vm::EagerBlobObjectList>(std::move(output_blobs));
  const auto& var_blob_list_ptr =
      std::make_shared<const vm::EagerBlobObjectList>(std::move(var_blobs));
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->LaunchLazyJob(input_blob_list_ptr, output_blob_list_ptr, var_blob_list_ptr,
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
