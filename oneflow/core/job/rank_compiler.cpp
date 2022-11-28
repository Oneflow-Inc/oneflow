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

namespace {

void CreateOpAttributeRef(Plan* plan, int64_t job_id, TaskProto* task_proto) {
  auto* job_id2op_attribute_ref_table = plan->mutable_job_id2op_attribute_ref_table();
  CHECK(task_proto->exec_sequence().exec_node_size() == 1);
  auto* exec_node = task_proto->mutable_exec_sequence()->mutable_exec_node(0);
  CHECK(exec_node->kernel_conf().has_op_attribute());
  const std::string op_name = exec_node->kernel_conf().op_attribute().op_conf().name();
  auto* op_name2op_attribute =
      (*job_id2op_attribute_ref_table)[job_id].mutable_op_name2op_attribute();
  auto find_it = op_name2op_attribute->find(op_name);
  if (find_it == op_name2op_attribute->end()) {
    op_name2op_attribute->insert(
        {op_name, task_proto->exec_sequence().exec_node(0).kernel_conf().op_attribute()});
  }
  auto* kernel_conf =
      task_proto->mutable_exec_sequence()->mutable_exec_node(0)->mutable_kernel_conf();
  kernel_conf->set_op_attribute_ref(op_name);
  // NOTE(levi): memory of op_attribute_ is released here.
  kernel_conf->set_allocated_op_attribute(nullptr);
}

LazyInitRegstDescIdProvider* GetLazyInitRegstDescIdProvider(RegstDesc* regst_desc) {
  auto* provider = regst_desc->mut_regst_desc_id_provider();
  auto* regst_desc_id_provider = dynamic_cast<LazyInitRegstDescIdProvider*>(provider);
  return CHECK_NOTNULL(regst_desc_id_provider);
}

void TryInitProducedRegstDescIdByNewRegstDescId(TaskNode* task_node) {
  task_node->ForEachProducedRegst([&](const std::string& _, RegstDesc* regst_desc) {
    auto* regst_desc_id_provider = GetLazyInitRegstDescIdProvider(regst_desc);
    if (regst_desc_id_provider->has_regst_desc_id()) { return; }
    regst_desc_id_provider->init_regst_desc_id();
  });
}

void InitProducedRegstDescId(TaskNode* task_node) {
  const HashMap<RegstDesc*, int64_t> regst_desc2predefined_regst_desc_id = [&] {
    HashMap<RegstDesc*, int64_t> regst_desc2predefined_regst_desc_id;
    for (const auto* edge : task_node->out_edges()) {
      auto* comm_task_node = dynamic_cast<CopyCommNetTaskNode*>(edge->dst_node());
      if (comm_task_node == nullptr) { continue; }
      RegstDesc* sole_regst_desc = nullptr;
      edge->ForEachRegstDesc([&](RegstDesc* regst_desc) {
        CHECK(sole_regst_desc == nullptr);
        sole_regst_desc = regst_desc;
      });
      auto* predefined = &regst_desc2predefined_regst_desc_id[sole_regst_desc];
      *predefined = std::max(*predefined, comm_task_node->candidate_in_regst_desc_id());
    }
    return regst_desc2predefined_regst_desc_id;
  }();
  // Initialize RegstDesc.regst_desc_id() by predefined regst_desc_id.
  for (const auto& pair : regst_desc2predefined_regst_desc_id) {
    auto* regst_desc_id_provider = GetLazyInitRegstDescIdProvider(pair.first);
    CHECK(!regst_desc_id_provider->has_regst_desc_id());
    regst_desc_id_provider->init_regst_desc_id(pair.second);
  }
  // Initialize RegstDesc.regst_desc_id() by new regst_desc_id.
  TryInitProducedRegstDescIdByNewRegstDescId(task_node);
}

}  // namespace

Maybe<void> RankCompiler::Compile(const HashSet<std::string>& var_op_names, Job* job,
                                  Plan* plan) const {
  // build task_gph.
  // TODO(levi): we can rewrite this part of code in visitor pattern.
  auto task_gph = JUST(RankTaskGraph::New(boxing_task_graph_proto_, var_op_names, rank_));
  using std::placeholders::_1;
  const auto& IsNotMyDuty = [&](const CompTaskNode* comp_task_node) {
    if (comp_task_node == nullptr) { return false; }
    const auto& parallel_desc = comp_task_node->op_node()->parallel_desc();
    return !task_gph->IsDutyRank(parallel_desc, comp_task_node->machine_id());
  };
  task_gph->ForEachNode(std::bind(&TaskNode::ProduceAllRegstsAndBindEdges, _1));
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
  // Initialize RegstDesc::regst_desc_id().
  task_gph->ForEachNode(&InitProducedRegstDescId);
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
    TryInitProducedRegstDescIdByNewRegstDescId(task_node);
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
      CreateOpAttributeRef(plan, job_desc.job_id(), &task_proto);
    }
    plan->mutable_task()->Add(std::move(task_proto));
  });
  // NOTE(levi): release task_gph here to decrise memory peak.
  task_gph.reset();

  // post-process for plan and delete Singleton<OpGraph>.
  auto* job_id2job_conf = plan->mutable_job_confs()->mutable_job_id2job_conf();
  (*job_id2job_conf)[GlobalJobDesc().job_id()] = GlobalJobDesc().job_conf();
  // NOTE(chengcheng): infer mem blob id & set inplace & add ctrl
  // TODO(chengcheng): set inplace hint for cpu regst
  IntraJobMemSharingUtil::InferMemBlockId4MemReusedRegst(plan, IsReachable);
  PlanUtil::MergeMemBlockIdByLogicalChainId(plan, *job);
  PlanUtil::SetUniqueMemBlockId4UnreusedMemRegst(plan);
  PlanUtil::SetForceInplaceMemBlock(plan);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
