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
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/job/runtime.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

NNGraph::~NNGraph() { runtime_.reset(); }

const std::vector<std::string>& NNGraph::inputs_op_names() const { return input_op_names_; }

const std::vector<std::string>& NNGraph::outputs_op_names() const { return output_op_names_; }

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
  // TODO(chengcheng): BufferMgr->NewBuffer for each inputs, outputs, wait ids, callback.
  runtime_.reset(new Runtime(plan_, GetMaxVal<size_t>(), false));
  return Maybe<void>::Ok();
}

Maybe<void> RunLazyNNGraph(const std::vector<std::shared_ptr<one::Tensor>>& inputs,
                           const std::vector<std::shared_ptr<one::Tensor>>& outputs,
                           const std::vector<std::shared_ptr<one::Tensor>>& parameters,
                           const std::shared_ptr<NNGraph>& nn_graph) {
  TODO();  // call InstructionsBuilder::RunLazyJob
  return Maybe<void>::Ok();
}

}  // namespace oneflow
