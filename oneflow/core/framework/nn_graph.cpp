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
#include <cstdint>
#include <memory>
#include <string>
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/time_util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/job/critical_section_instance.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/job_rewriter/job_completer.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/rpc/include/base.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
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

REGISTER_FUNCTION_CONFIG_DEF().Bool("__is_user_function__", true, "is user defined function");

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

Maybe<void> NNGraph::RangePushPlan(Plan* global_plan, const std::string& plan_name_prefix, int64_t start_rank, int64_t rank_range_size) {
  const int64_t end_rank = start_rank + rank_range_size - 1;
  const int64_t cpu_num = std::thread::hardware_concurrency();
  const int64_t thread_pool_size = std::min(rank_range_size, cpu_num);
  ThreadPool thread_pool(thread_pool_size);
  BlockingCounter counter(rank_range_size);
  for (int64_t rank_id = start_rank; rank_id <= end_rank ; ++rank_id) {
    thread_pool.AddWork([this, rank_id, &global_plan, &plan_name_prefix, &counter]() {
      std::string rank_plan_name = plan_name_prefix + std::to_string(rank_id);
      // Creat sub-plan.
      auto tc_sub_plan = std::make_unique<TimeCounter<std::chrono::milliseconds>>(true);
      Plan sub_plan;
      sub_plan.set_allocated_job_confs(global_plan->mutable_job_confs());
      tc_sub_plan->Count(rank_plan_name + " add job conf", 1);

      sub_plan.set_allocated_collective_boxing_plan(global_plan->mutable_collective_boxing_plan());
      tc_sub_plan->Count(rank_plan_name + " add collective boxing", 1);

      sub_plan.set_allocated_ctrl_regst_desc_info(global_plan->mutable_ctrl_regst_desc_info());
      tc_sub_plan->Count(rank_plan_name + " add ctrl regst", 1);

	    // TODO(strint): rm copy.
      for (auto& pair : *global_plan->mutable_job_id2op_attribute_ref_table()) {
        sub_plan.mutable_job_id2op_attribute_ref_table()->insert(pair);
      }
      tc_sub_plan->Count(rank_plan_name + " add op attr", 1);

      for (auto& task_proto : *global_plan->mutable_task()) {
        if (task_proto.machine_id() == rank_id) {
          sub_plan.mutable_task()->AddAllocated(&task_proto);
        }
      }
      tc_sub_plan->Count(rank_plan_name + " add task", 1);

      for (auto& mem_block_proto : *global_plan->mutable_block_chunk_list()->mutable_mem_block()) {
        if (mem_block_proto.machine_id() == rank_id) {
          sub_plan.mutable_block_chunk_list()->mutable_mem_block()->AddAllocated(&mem_block_proto);
        }
      }
      tc_sub_plan->Count(rank_plan_name + " add mem block", 1);

      for (auto& chunk_proto : *global_plan->mutable_block_chunk_list()->mutable_chunk()) {
        if (chunk_proto.machine_id() == rank_id) {
          sub_plan.mutable_block_chunk_list()->mutable_chunk()->AddAllocated(&chunk_proto);
        }
      }
      tc_sub_plan->Count(rank_plan_name + " add chunk", 1);

      if (rank_id == 0) {
        // sub_plan used zero copy, so here needs copy.
        plan_.CopyFrom(sub_plan);
      } else {
        Singleton<CtrlClient>::Get()->PushMasterKV(rank_plan_name, sub_plan);
        tc_sub_plan->Count(rank_plan_name + " PushKV", 1);
        VLOG(1) << "[elapsed]rank id " << GlobalProcessCtx::Rank() << " push plan " << rank_plan_name << " size " << sub_plan.ByteSizeLong();
      }
      // Set allocated needs to realease ownership to avoid double free.
      sub_plan.release_job_confs();
      sub_plan.release_collective_boxing_plan();
      sub_plan.release_ctrl_regst_desc_info();
      while(!sub_plan.task().empty()) {
        sub_plan.mutable_task()->ReleaseLast();
      }
      while(!sub_plan.block_chunk_list().mem_block().empty()) {
        sub_plan.mutable_block_chunk_list()->mutable_mem_block()->ReleaseLast();
      }
      while(!sub_plan.block_chunk_list().chunk().empty()) {
        sub_plan.mutable_block_chunk_list()->mutable_chunk()->ReleaseLast();
      }
      counter.Decrease();
    });
  }
  // Wait for all sub plan in range has finished.
  counter.WaitForeverUntilCntEqualZero();
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::RangePullPlan(const std::string& plan_name_prefix, int64_t start_rank, int64_t rank_range_size) {
  const int64_t this_rank_id = GlobalProcessCtx::Rank();
  if (this_rank_id >= start_rank && this_rank_id < start_rank + rank_range_size && this_rank_id != 0) {
    std::string rank_plan_name = plan_name_prefix + std::to_string(this_rank_id);
    Singleton<CtrlClient>::Get()->PullMasterKV(rank_plan_name, &plan_);
  }
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::RangeClearPlan(const std::string& plan_name_prefix, int64_t start_rank, int64_t rank_range_size) {
  for (int64_t rank_id = start_rank; rank_id < start_rank + rank_range_size; ++rank_id) {
    if (rank_id == 0) { continue; }
    std::string rank_plan_name = plan_name_prefix + std::to_string(rank_id);
    Singleton<CtrlClient>::Get()->ClearMasterKV(rank_plan_name);
  }
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::DeleteOutdatedVariableInVariableTensorMgr() {
  std::set<std::string> variable_names = *JUST([&]() -> Maybe<std::set<std::string>> {
    std::set<std::string> variable_names_;
    OpGraph op_graph(job_);
    JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
      if (op_node->op().op_conf().has_variable_conf() == false) { return Maybe<void>::Ok(); }
      variable_names_.insert(op_node->op().op_name());
      return Maybe<void>::Ok();
    }));
    return variable_names_;
  }());

  auto mgr = Singleton<VariableTensorMgr>::Get();
  for (auto& name : mgr->DumpNames()) {
    if (variable_names.find(name) == variable_names.end()) { mgr->Delete(name); }
  }
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::CompileAndInitRuntime() {
  CHECK_OR_RETURN(!runtime_inited_);
  auto compile_tc = std::make_unique<TimeCounter<std::chrono::seconds>>(true);
  auto tc = std::make_unique<TimeCounter<std::chrono::milliseconds>>(true);
  JUST(RegisterFreeEagerTensorsToVariableOpNames());
  tc->Count("Graph name: " + name_ + " RegisterFreeEagerTensorsToVariableOpNames", 1);
  JUST(RegisterNewVariableOpInJobPass());
  tc->Count("Graph name: " + name_ + " RegisterNewVariableOpInJobPass", 1);
  JUST(DeleteOutdatedVariableInVariableTensorMgr());
  tc->Count("Graph name: " + name_ + " DeleteOutdatedVariableInVariableTensorMgr", 1);

  // NOTE(chengcheng): TensorNameScope need to be cleared after current graph is built.
  one::TensorNameScope::Global()->Clear();
  // Clear all backward pass scope
  ClearAllBackwardPassScope();

  // NOTE(chengcheng): Singleton<JobDesc> need be clear before GlobalJobDescScope construct.
  if (Singleton<JobDesc>::Get() != nullptr) { Singleton<JobDesc>::Delete(); }

  auto scope = std::make_unique<GlobalJobDescScope>(job_.job_conf(), job_id_);

  // NOTE(chengcheng): do job compeleter for each rank.
  JUST(JobCompleter().Complete(&job_));
  tc->Count("Graph name: " + name_ + " Complete job", 1);

  Plan global_plan;
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    // TODO(chengcheng): new memory reused by chunk
    std::shared_ptr<TaskGraph> task_graph;
    PlanCompiler::Compile(&job_, &global_plan, task_graph);
    CHECK_OR_RETURN(task_graph);
    tc->Count("Graph name: " + name_ + " Compile plan", 1);
    PlanUtil::GenMemBlockAndChunkWithVariableOpNames4Plan(&global_plan, std::const_pointer_cast<const TaskGraph>(task_graph), variable_op_names_);
    tc->Count("Graph name: " + name_ + " Generate MemBlock and Chunk", 1);

    PlanUtil::GenRegisterHint(&global_plan);
    tc->Count("Graph name: " + name_ + " GenRegisterHint", 1);
    // TODO(chengcheng): test collective boxing for multi-job.
    PlanUtil::GenCollectiveBoxingPlan(&job_, std::const_pointer_cast<const TaskGraph>(task_graph), &global_plan);
    tc->Count("Graph name: " + name_ + " GenCollectiveBoxingPlan", 1);
    // PlanUtil::SetForceInplaceMemBlock(&plan_); NOTE(chengcheng): only for ssp.
    PlanUtil::DumpCtrlRegstInfoToPlan(&global_plan);
    tc->Count("Graph name: " + name_ + " DumpCtrlRegstInfoToPlan", 1);
    PlanUtil::PlanMemoryLog(&global_plan, name_);
    if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
      PlanUtil::GenLightPlan(&global_plan, name_);
    }
    tc->Count("Graph name: " + name_ + " Memory and Plan Log", 1);

    // NOTE(strint): Add op attr into plan.
    {
      const int64_t node_num = task_graph->node_num();
      const int64_t cpu_num = std::thread::hardware_concurrency();
      const int64_t thread_pool_size = std::min(node_num, cpu_num);
      BlockingCounter counter(node_num);
      std::mutex mtx;
      ThreadPool thread_pool(thread_pool_size);

      auto* job_id2op_attribute_ref_table = global_plan.mutable_job_id2op_attribute_ref_table();
      auto* op_name2op_attribute =
          (*job_id2op_attribute_ref_table)[job_id_].mutable_op_name2op_attribute();
      task_graph->ForEachNode([&](TaskNode* task_node) {
        thread_pool.AddWork([task_node, op_name2op_attribute, &counter, &mtx]() {
          if (!task_node->IsMeaningLess() && task_node->op_node()) {
              auto op_node = task_node->op_node();
              const std::string op_name = op_node->op().op_name();
              {
                std::unique_lock<std::mutex> guard(mtx);
                auto find_it = op_name2op_attribute->find(op_name);
                if (find_it == op_name2op_attribute->end()) {
                  OpAttribute op_attr;
                  CHECK_JUST(op_node->op().ToOpAttribute(&op_attr));
                  // TODO(strint): Try to optimize here
                  op_name2op_attribute->insert({op_name, op_attr});
                }  // guard(mtx)
              }
          }
          counter.Decrease();
        } /* thread_pool.AddWork */);
      } /* task_gph->ForEachNode */);
      counter.WaitForeverUntilCntEqualZero();
      tc->Count("Graph name: " + name_ + " AddOpAttrtoPlan", 1);
    }
    if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
      TeePersistentLogStream::Create("job_" + name_ + "_plan")->Write(global_plan);
      PlanUtil::ToDotFile(global_plan, task_graph, "job_" + name_ + "_plan.dot");
      tc->Count("Graph name: " + name_ + " LogPlan", 1);
    }
    VLOG(1) << "[elapsed]rank id " << GlobalProcessCtx::Rank() << " global plan size " << global_plan.ByteSizeLong();
  }
  tc->Count("Graph name: " + name_ + " ReleaseTaskGraph", 1);
  compile_tc->Count("Graph name: " + name_ + " TotalCompile", 1);
  if (GlobalProcessCtx::WorldSize() == 1) {
    plan_.Swap(&global_plan);
  } else if (GlobalProcessCtx::WorldSize() > 1) {
    std::string plan_name_prefix = "plan_" + job_name() + "_r_";
    int64_t rang_push_size = 2, cur_start_rank = 0, cur_range_size = 0;
    while (cur_start_rank < GlobalProcessCtx::WorldSize()) {
      cur_range_size = std::min(rang_push_size, static_cast<int64_t>(GlobalProcessCtx::WorldSize() - cur_start_rank));
      // Use range push to limit memory consumption.
      if (GlobalProcessCtx::IsThisProcessMaster()) {
        JUST(RangePushPlan(&global_plan, plan_name_prefix, cur_start_rank, cur_range_size));
        tc->Count("Graph name: " + name_ + " PushPlan" + std::to_string(cur_start_rank) + "to" + std::to_string(cur_start_rank + cur_range_size - 1), 1);
      } else {
        JUST(RangePullPlan(plan_name_prefix, cur_start_rank, cur_range_size));
        tc->Count("Graph name: " + name_ + " PullPlan" + std::to_string(cur_start_rank) + "to" + std::to_string(cur_start_rank + cur_range_size - 1), 1);
      }
      // Sync to make sure this range of sub plans has been received.
      OF_SESSION_BARRIER();
      tc->Count("Graph name: " + name_ + " FinishPushSubPlan" + std::to_string(cur_start_rank) + "to" + std::to_string(cur_start_rank + cur_range_size - 1), 1);
      if (GlobalProcessCtx::IsThisProcessMaster()) {
        JUST(RangeClearPlan(plan_name_prefix, cur_start_rank, cur_range_size));
      }
      tc->Count("Graph name: " + name_ + " FinishClearSubPlan", 1);
      cur_start_rank += cur_range_size;
    }
    global_plan.Clear();
    tc->Count("Graph name: " + name_ + " ClearGlobalPlan", 1);
  }

  // NOTE(chengcheng): recovery op_attr
  PlanUtil::PopulateOpAttribute(&plan_, plan_.mutable_job_id2op_attribute_ref_table());
  tc->Count("Graph name: " + name_ + " PopulateOpAttribute", 1);
  compile_tc->Count("Graph name: " + name_ + " PlanSync", 1);
  CHECK_OR_RETURN(false);

  NewRuntimeBuffers();
  tc->Count("Graph name: " + name_ + " NewRuntimeBuffers", 1);

  // There maybe some difference of logical graph between difference rank.
  // After plan synchronization, all ranks get the same plan.
  // Then the plan can be used to create new variable.
  JUST(GetVariableRealBlobAfterSyncPlan());
  tc->Count("Graph name: " + name_ + " GetVariableRealBlobAfterSyncPlan", 1);

  // NOTE(strint): Do memory shrink to free cached memory in eager VM before graph runtime init.
  JUST(vm::CurrentRankSync());
  auto* vm = JUST(SingletonMaybe<VirtualMachine>());
  JUST(vm->ShrinkAllMem());
  tc->Count("Graph name: " + name_ + " VM::ShrinkAllMem", 1);

  // Start graph runtime.
  runtime_.reset(new Runtime(plan_, variable_op_name2eager_blob_object_));
  tc->Count("Graph name: " + name_ + " RuntimeInit", 1);
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
