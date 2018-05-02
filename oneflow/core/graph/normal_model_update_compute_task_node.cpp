#include "oneflow/core/graph/normal_model_update_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"

namespace oneflow {

void NormalMdUpdtCompTaskNode::ProduceAllRegstsAndBindEdges() {
  int32_t max_model_regst = -1;
  if (Global<JobDesc>::Get()->IsPredict()) {
    max_model_regst = 1;
  } else {
    if (Global<JobDesc>::Get()->Staleness() == -1) {
      max_model_regst = kMaxRegisterNum;
    } else {
      max_model_regst = Global<JobDesc>::Get()->Staleness() + 1;
    }
  }
  auto model_regst = ProduceRegst("model", 1, max_model_regst);
  auto model_tmp_regst = ProduceRegst("model_tmp", 1, 1);
  ProduceRegst("data_tmp", 1, 1);
  related_init_model_task_id_ = -1;
  for (TaskEdge* out_edge : out_edges()) {
    TaskNode* dst_node = out_edge->dst_node();
    if (IsForwardTaskType(dst_node->GetTaskType()) || IsBackwardTaskType(dst_node->GetTaskType())) {
      out_edge->AddRegst("model", model_regst);
      out_edge->AddRegst("model_tmp", model_tmp_regst);
      if (IsForwardTaskType(dst_node->GetTaskType()) && related_init_model_task_id_ == -1) {
        auto fw_node = static_cast<NormalForwardCompTaskNode*>(dst_node);
        fw_node->set_random_seed(random_seed_);
        related_init_model_task_id_ = fw_node->task_id();
      }
    } else {
      out_edge->AddRegst("model", model_regst);
    }
  }
}

void NormalMdUpdtCompTaskNode::ConsumeAllRegsts() {
  if (Global<JobDesc>::Get()->IsPredict()) { return; }
  for (TaskEdge* edge : in_edges()) {
    ConsumeRegst("model_diff_acc_" + NewUniqueId(), edge->GetSoleRegst());
  }
}

bool NormalMdUpdtCompTaskNode::IsReadyForBuild() {
  return GetProducedRegst("model")->IsLocked() && GetProducedRegst("model_tmp")->IsLocked();
}

void NormalMdUpdtCompTaskNode::BuildExecGphAndRegst() {
  if (Global<JobDesc>::Get()->IsPredict()) { return; }
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  size_t ibn_idx = 0;
  for (const auto& pair : consumed_regsts()) {
    node->BindBnWithRegst(node->op()->input_bns().Get(ibn_idx++), pair.second.front().lock());
  }
  node->BindBnWithRegst(node->op()->SoleObn(), GetProducedRegst("model"));
  node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, GetProducedRegst("data_tmp"));
  node->InferBlobDescs(nullptr);
}

void NormalMdUpdtCompTaskNode::LockRegsts() { GetProducedRegst("data_tmp")->Lock(); }

void NormalMdUpdtCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  ForEachNodeOnOutEdge([&](const TaskNode* node) {
    if (IsForwardTaskType(node->GetTaskType())) {
      // do nothing
    } else if (IsBackwardTaskType(node->GetTaskType())) {
      // do nothing
    } else {
      CHECK_EQ(task_proto->related_save_model_task_id(), -1);
      task_proto->set_related_save_model_task_id(node->task_id());
    }
  });
  task_proto->set_related_init_model_task_id(related_init_model_task_id_);
}

}  // namespace oneflow
