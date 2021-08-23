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
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/vm/vm_util.h"

namespace oneflow {

NNGraph::~NNGraph() {
  VLOG(2) << "graph destructor Try to close c nn graph name " << name_ << "." << std::endl;
  CHECK_JUST(Close());
}

Maybe<void> NNGraph::Close() {
  if (!is_closed_) {
    VLOG(2) << "Try to close c nn graph name " << name_ << "." << std::endl;
    CloseRuntimeBuffers();
    runtime_.reset();
    Global<MultiClientSessionContext>::Get()->RemoveGraphFreeEagerTensors(name_);
    is_closed_ = true;
    VLOG(2) << "Finish close c nn graph name " << name_ << "." << std::endl;
  }
  return Maybe<void>::Ok();
}

const std::vector<std::string>& NNGraph::inputs_op_names() const { return input_op_names_; }

const std::vector<std::string>& NNGraph::outputs_op_names() const { return output_op_names_; }

int64_t NNGraph::variable_op_size() const { return variable_op_name2eager_blob_.size(); }

Maybe<void> NNGraph::RegisterInputOpNames(const std::vector<std::string>& input_op_names) {
  input_op_names_.assign(input_op_names.begin(), input_op_names.end());
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::RegisterOutputOpNames(const std::vector<std::string>& output_op_names) {
  output_op_names_.assign(output_op_names.begin(), output_op_names.end());
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::RegisterVariableOpNamesAndTensors(
    const std::vector<std::string>& variable_op_names,
    const std::vector<std::shared_ptr<one::Tensor>>& variable_tensors) {
  JUST(vm::MultiClientSync());
  CHECK_EQ_OR_RETURN(variable_op_names.size(), variable_tensors.size());
  CHECK_OR_RETURN(variable_op_name2eager_blob_.empty());
  for (int32_t i = 0; i < variable_op_names.size(); ++i) {
    const std::shared_ptr<one::Tensor>& var = variable_tensors.at(i);
    CHECK_OR_RETURN(var->is_eager());
    Blob* var_blob = nullptr;
    if (var->is_consistent()) {
      // NOTE(chengcheng): var_blob maybe nullptr when consistent tensor has no cur_rank_phy_tensor.
      const std::shared_ptr<one::MirroredTensor> local_var = JUST(var->cur_rank_phy_tensor());
      var_blob = JUST(local_var->eager_blob_object())->mut_blob();
    } else {
      var_blob = JUST(var->eager_blob_object())->mut_blob();
    }
    const std::string& var_name = variable_op_names.at(i);
    CHECK_OR_RETURN(!var_name.empty());
    CHECK_OR_RETURN(variable_op_name2eager_blob_.emplace(var_name, var_blob).second);
    CHECK_OR_RETURN(variable_op_names_.insert(var_name).second);
  }
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::RegisterFreeEagerTensorsToVariableOpNames() {
  JUST(vm::MultiClientSync());
  const auto& free_eager_tensors =
      Global<MultiClientSessionContext>::Get()->GetFreeEagerTensorNamePairByGraphName(name_);
  for (const auto& pair : free_eager_tensors) {
    const std::string& var_name = pair.first;
    const std::shared_ptr<one::Tensor>& var = pair.second;
    CHECK_OR_RETURN(var->is_eager());
    Blob* var_blob = nullptr;
    if (var->is_consistent()) {
      const std::shared_ptr<one::MirroredTensor> local_var = JUST(var->cur_rank_phy_tensor());
      var_blob = JUST(local_var->eager_blob_object())->mut_blob();
    } else {
      var_blob = JUST(var->eager_blob_object())->mut_blob();
    }
    CHECK_OR_RETURN(!var_name.empty());
    CHECK_OR_RETURN(variable_op_name2eager_blob_.emplace(var_name, var_blob).second);
    CHECK_OR_RETURN(variable_op_names_.insert(var_name).second);
  }
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::CompileAndInitRuntime() {
  JUST(RegisterFreeEagerTensorsToVariableOpNames());
  CHECK_OR_RETURN(!runtime_inited_);
  JobBuildAndInferCtx* job_ctx = JUST(GetJobBuildAndInferCtx(name_));
  job_ = job_ctx->job();
  // TODO(chengcheng): CHECK job valid for each rank.

  // NOTE(chengcheng): Global<JobDesc> need be clear before GlobalJobDescScope construct.
  if (Global<JobDesc>::Get() != nullptr) { Global<JobDesc>::Delete(); }

  auto scope = std::make_unique<GlobalJobDescScope>(job_.job_conf(), job_ctx->job_id());
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    double start = GetCurTime();
    // TODO(chengcheng): new memory reused by chunk
    Compiler().Compile(&job_, &plan_, /* need_job_complete */ true);
    PlanUtil::GenMemBlockAndChunkWithVariableOpNames4Plan(&plan_, variable_op_names_);

    LOG(INFO) << "\njob_id: " << job_ctx->job_id() << " , job_name: " << name_
              << " , compile time: " << (GetCurTime() - start) / 1000000000.0 << " seconds.\n";
    if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
      TeePersistentLogStream::Create("job_" + name_ + "_plan")->Write(plan_);
    }
    // TODO(chengcheng): test collective boxing for multi-job.
    PlanUtil::GenCollectiveBoxingPlan(&job_, &plan_);
    // PlanUtil::SetForceInplaceMemBlock(&plan_); NOTE(chengcheng): only for ssp.
    PlanUtil::DumpCtrlRegstInfoToPlan(&plan_);
  }
  if (GlobalProcessCtx::WorldSize() > 1) {
    std::string plan_name = "plan:" + job_name();
    if (GlobalProcessCtx::IsThisProcessMaster()) {
      // TODO(chengcheng): split plan for each rank.
      Global<CtrlClient>::Get()->PushKV(plan_name, plan_);
    } else {
      Global<CtrlClient>::Get()->PullKV(plan_name, &plan_);
    }
    OF_SESSION_BARRIER();
    // NOTE(zwx): After barrier plan is synchronized between all ranks,
    //     then it can be cleared for saving mem.
    if (GlobalProcessCtx::IsThisProcessMaster()) { Global<CtrlClient>::Get()->ClearKV(plan_name); }
  }
  // NOTE(chengcheng): recovery op_attr
  PlanUtil::PopulateOpAttibute(&plan_, plan_.job_id2op_attribute_ref_table());

  NewRuntimeBuffers();
  runtime_.reset(new Runtime(plan_, variable_op_name2eager_blob_));
  runtime_inited_ = true;
  return Maybe<void>::Ok();
}

void NNGraph::NewRuntimeBuffers() {
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
  // NOTE(chengcheng):
  //   The BufferSize for each Buffer:
  //   1. SourceTick and CallbackNotifier is job_conf.concurrency_width by user (default = 128)
  //     in Pipeline Parallelism, this value need greater than pipeline stage num for pipelining.
  //   2. Input/Output Buffer is 2 because this is the minimum size of pipeline async launch job.
  size_t concurrency_width = job_.job_conf().concurrency_width();
  buffer_mgr->NewBuffer(GetSourceTickBufferName(name_), concurrency_width);
  buffer_mgr->NewBuffer(GetCallbackNotifierBufferName(name_), concurrency_width);
  for (const std::string& input_op_name : input_op_names_) {
    buffer_mgr->NewBuffer(GetInputBufferName(name_, input_op_name), 2);
  }
  for (const std::string& output_op_name : output_op_names_) {
    buffer_mgr->NewBuffer(GetOutputBufferName(name_, output_op_name), 2);
  }
}

void NNGraph::CloseRuntimeBuffers() {
  if (runtime_inited_) {
    auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
    for (const std::string& output_op_name : output_op_names_) {
      buffer_mgr->Get(GetOutputBufferName(name_, output_op_name))->Close();
    }
    for (const std::string& input_op_name : input_op_names_) {
      buffer_mgr->Get(GetInputBufferName(name_, input_op_name))->Close();
    }
    buffer_mgr->Get(GetCallbackNotifierBufferName(name_))->Close();
    buffer_mgr->Get(GetSourceTickBufferName(name_))->Close();
  }
}

namespace {

Maybe<void> MakeEagerBlobObjectList(std::vector<std::shared_ptr<vm::EagerBlobObject>>* blob_list,
                                    const one::TensorTuple& tensor_list) {
  for (const auto& tensor : tensor_list) {
    CHECK_OR_RETURN(tensor->is_eager());
    if (tensor->is_consistent()) {
      blob_list->push_back(JUST(JUST(tensor->cur_rank_phy_tensor())->eager_blob_object()));
    } else {
      blob_list->push_back(JUST(tensor->eager_blob_object()));
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
  //   parameters not used in RunLazyJobInstrucntion;
  //   the args: parameters is all variable tensor hold by nn.Graph
  //   but the NNGraph::variable_op_size may has FreeEagerTensor as sepcial variable op.
  CHECK_LE_OR_RETURN(parameters.size(), nn_graph->variable_op_size());
  std::vector<std::shared_ptr<vm::EagerBlobObject>> input_blobs;
  std::vector<std::shared_ptr<vm::EagerBlobObject>> output_blobs;
  std::vector<std::shared_ptr<vm::EagerBlobObject>> var_blobs;
  JUST(MakeEagerBlobObjectList(&input_blobs, inputs));
  JUST(MakeEagerBlobObjectList(&output_blobs, outputs));
  JUST(MakeEagerBlobObjectList(&var_blobs, parameters));
  const auto& input_blob_list_ptr =
      std::make_shared<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>(
          std::move(input_blobs));
  const auto& output_blob_list_ptr =
      std::make_shared<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>(
          std::move(output_blobs));
  const auto& var_blob_list_ptr =
      std::make_shared<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>(
          std::move(var_blobs));
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->RunLazyJob(input_blob_list_ptr, output_blob_list_ptr, var_blob_list_ptr,
                               nn_graph);
  }));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
