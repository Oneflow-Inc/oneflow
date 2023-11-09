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
#include "oneflow/core/job/rank_compiler.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/intra_job_mem_sharing_util.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job_rewriter/job_completer.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

Maybe<void> RankCompiler::Compile(const HashSet<std::string>& var_op_names, Job* job,
                                  Plan* plan) const {
#ifdef WITH_CUDA
  // Use the right device when some plan compilation needs cuda to avoid creating unnecessary cuda
  // context on cuda:0.
  CudaCurrentDeviceGuard guard(GetCudaDeviceIndex());
#endif  // WITH_CUDA
  auto task_gph = JUST(RankTaskGraph::New(boxing_task_graph_proto_, var_op_names, rank_));
  using std::placeholders::_1;
  const auto& IsNotMyDuty = [&](const CompTaskNode* comp_task_node) {
    if (comp_task_node == nullptr) { return false; }
    const auto& parallel_desc = comp_task_node->op_node()->parallel_desc();
    return !task_gph->IsDutyRank(parallel_desc, comp_task_node->machine_id());
  };
  task_gph->ForEachNode([&](TaskNode* task_node) {
    auto* comp_task_node = dynamic_cast<CompTaskNode*>(task_node);
    if (IsNotMyDuty(comp_task_node)) {
      auto* fake_consumed_regsts_provider =
          dynamic_cast<FakeConsumedRegstProvider*>(comp_task_node);
      CHECK_NOTNULL(fake_consumed_regsts_provider)->ConsumeFakeRegstsIf();
    } else {
      task_node->ConsumeAllRegsts();
    }
  });
  task_gph->ForEachNode([&](TaskNode* task_node) {
    auto* comp_task_node = dynamic_cast<CompTaskNode*>(task_node);
    if (IsNotMyDuty(comp_task_node)) {
      // Do nothing. because all consumed registers are fake.
    } else {
      task_node->PinConsumedRegst();
    }
  });
  task_gph->TopoForEachNode(&TaskNode::Build);
  task_gph->RemoveEmptyRegsts();
  task_gph->TopoForEachNode(&TaskNode::InferTimeShapeIfMeaningful);
  task_gph->DecideExecutionOrder();
  task_gph->MergeChainAndAddOrderingCtrlEdgeInSameChain();
  auto IsReachable = Singleton<OpGraph>::Get()->MakePredicatorIsOpNameDataOrCtrlReachable();
  const JobDesc& job_desc = GlobalJobDesc();
  if (job_desc.enable_inplace()) {
    task_gph->ForEachGpuDeviceNodes([&](const HashSet<TaskNode*>& dev_nodes) {
      if (dev_nodes.empty()) { return; }
      if ((*dev_nodes.begin())->machine_id() != rank_) { return; }  // other ranks are ignored.
      task_gph->EnableInplaceMemSharing(dev_nodes, IsReachable);
    });
  }
  task_gph->ForEachEdge([&](TaskEdge* task_edge) { task_edge->CheckRegstLbiValid(); });

  // put infomation from task_gph into plan.
  task_gph->ForEachNode([&](TaskNode* task_node) {
    if (task_node->IsMeaningLess()) { return; }
    auto* comp_task_node = dynamic_cast<CompTaskNode*>(task_node);
    if (comp_task_node != nullptr) {
      const auto& parallel_desc = comp_task_node->op_node()->parallel_desc();
      if (!task_gph->IsDutyRank(parallel_desc, task_node->machine_id())) {
        auto* fake_consumed_regsts_provider =
            dynamic_cast<FakeConsumedRegstProvider*>(comp_task_node);
        CHECK_NOTNULL(fake_consumed_regsts_provider)->EraseFakeRegstsIf();
      }
    }
    TaskProto task_proto;
    task_node->ToProto(&task_proto);
    if (task_node->GetTaskType() == kNormalForward || task_node->GetTaskType() == kRepeat
        || task_node->GetTaskType() == kAcc) {
      PlanUtil::CreateOpAttributeRef(plan, job_desc.job_id(), &task_proto);
    }
    plan->mutable_task()->Add(std::move(task_proto));
  });

  // post-process for plan and delete Singleton<OpGraph>.
  auto* job_id2job_conf = plan->mutable_job_confs()->mutable_job_id2job_conf();
  (*job_id2job_conf)[GlobalJobDesc().job_id()] = GlobalJobDesc().job_conf();
  // NOTE(chengcheng): infer mem blob id & set inplace & add ctrl
  IntraJobMemSharingUtil::InferMemBlockId4MemReusedRegst(plan);
  PlanUtil::MergeMemBlockIdByLogicalChainId(plan, *job, rank_);
  PlanUtil::SetUniqueMemBlockId4UnreusedMemRegst(plan);
  PlanUtil::SetForceInplaceMemBlock(plan, rank_);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
