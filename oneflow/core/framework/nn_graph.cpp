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
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/job/runtime.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

NNGraph::~NNGraph() {
  CloseRuntimeBuffers();
  runtime_.reset();
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
  CHECK_EQ_OR_RETURN(variable_op_names.size(), variable_tensors.size());
  CHECK_OR_RETURN(variable_op_name2eager_blob_.empty());
  for (int32_t i = 0; i < variable_op_names.size(); ++i) {
    const std::shared_ptr<one::Tensor>& var = variable_tensors.at(i);
    CHECK_OR_RETURN(var->is_eager());
    Blob* var_blob = nullptr;
    if (var->is_consistent()) {
      // TODO(chengcheng): handle for consistent variable which has NO phy tensor in cur rank.
      const std::shared_ptr<one::MirroredTensor> local_var = JUST(var->cur_rank_phy_tensor());
      var_blob = JUST(local_var->eager_blob_object())->mut_blob();
    } else {
      var_blob = JUST(var->eager_blob_object())->mut_blob();
    }
    CHECK_OR_RETURN(variable_op_name2eager_blob_.emplace(variable_op_names.at(i), var_blob).second);
  }
  return Maybe<void>::Ok();
}

Maybe<void> NNGraph::CompileAndInitRuntime() {
  CHECK_OR_RETURN(!runtime_inited_);
  JobBuildAndInferCtx* job_ctx = JUST(GetJobBuildAndInferCtx(name_));
  job_ = job_ctx->job();
  // TODO(chengcheng): CHECK job valid for each rank.

  auto scope = std::make_unique<GlobalJobDescScope>(job_.job_conf(), job_ctx->job_id());
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    double start = GetCurTime();
    // TODO(chengcheng): new memory reused by chunk
    Compiler().Compile(&job_, &plan_, /* need_job_complete */ true);

    LOG(INFO) << "\njob_id: " << job_ctx->job_id() << " , job_name: " << name_
              << " , compile time: " << (GetCurTime() - start) / 1000000000.0 << " seconds.\n";
    if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
      TeePersistentLogStream::Create("job_" + name_ + "_plan")->Write(plan_);
    }
    // TODO(chengcheng): test collective boxing for multi-job.
    PlanUtil::GenCollectiveBoxingPlan(&job_, &plan_);
    PlanUtil::SetForceInplaceMemBlock(&plan_);
    PlanUtil::DumpCtrlRegstInfoToPlan(&plan_);
  }
  if (GlobalProcessCtx::WorldSize() > 1) {
    Global<CtrlClient>::Get()->ClearKV("plan");
    if (GlobalProcessCtx::IsThisProcessMaster()) {
      // TODO(chengcheng): split plan for each rank.
      Global<CtrlClient>::Get()->PushKV("plan", plan_);
    } else {
      Global<CtrlClient>::Get()->PullKV("plan", &plan_);
    }
    OF_SESSION_BARRIER();
  }
  NewRuntimeBuffers();
  runtime_.reset(new Runtime(plan_, GetMaxVal<size_t>(), false));
  runtime_inited_ = true;
  return Maybe<void>::Ok();
}

void NNGraph::NewRuntimeBuffers() {
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
  buffer_mgr->NewBuffer(GetSourceTickBufferName(name_), 128);
  buffer_mgr->NewBuffer(GetCallbackNotifierBufferName(name_), 128);
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
                                    const std::vector<std::shared_ptr<one::Tensor>>& tensor_list) {
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

Maybe<void> RunLazyNNGraph(const std::vector<std::shared_ptr<one::Tensor>>& inputs,
                           const std::vector<std::shared_ptr<one::Tensor>>& outputs,
                           const std::vector<std::shared_ptr<one::Tensor>>& parameters,
                           const std::shared_ptr<NNGraph>& nn_graph) {
  CHECK_EQ_OR_RETURN(inputs.size(), nn_graph->inputs_op_names().size());
  CHECK_EQ_OR_RETURN(outputs.size(), nn_graph->outputs_op_names().size());
  CHECK_EQ_OR_RETURN(parameters.size(), nn_graph->variable_op_size());
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
