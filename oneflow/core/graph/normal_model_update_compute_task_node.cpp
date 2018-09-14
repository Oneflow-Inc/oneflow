#include "oneflow/core/graph/normal_model_update_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

const NormalForwardCompTaskNode* NormalMdUpdtCompTaskNode::GetForwardTaskNode() const {
  for (const TaskEdge* out_edge : out_edges()) {
    const TaskNode* dst_node = out_edge->dst_node();
    if (IsForwardTaskType(dst_node->GetTaskType())) {
      return dynamic_cast<const NormalForwardCompTaskNode*>(dst_node);
    }
  }
  UNIMPLEMENTED();
}

bool NormalMdUpdtCompTaskNode::IsTrainable() const {
  return Global<JobDesc>::Get()->IsTrain()
         && GetForwardTaskNode()->logical_node()->SoleOp()->op_conf().trainable();
}

void NormalMdUpdtCompTaskNode::ProduceAllRegstsAndBindEdges() {
  int32_t max_model_regst = 1;
  auto model_regst = ProduceRegst("model", false, 1, max_model_regst);
  auto const_model_regst = ProduceRegst("const_model", false, 1, 1);
  auto moving_model_regst = ProduceRegst("moving_model", false, 1, 1);
  ProduceRegst("data_tmp", false, 1, 1);
  related_init_model_task_id_ = -1;
  for (TaskEdge* out_edge : out_edges()) {
    TaskNode* dst_node = out_edge->dst_node();
    if (IsForwardTaskType(dst_node->GetTaskType()) || IsBackwardTaskType(dst_node->GetTaskType())) {
      out_edge->AddRegst("model", model_regst);
      out_edge->AddRegst("const_model", const_model_regst);
      if (IsForwardTaskType(dst_node->GetTaskType()) && related_init_model_task_id_ == -1) {
        auto fw_node = static_cast<NormalForwardCompTaskNode*>(dst_node);
        fw_node->set_random_seed(random_seed_);
        related_init_model_task_id_ = fw_node->task_id();
      }
    } else {
      out_edge->AddRegst("model", model_regst);
      out_edge->AddRegst("moving_model", moving_model_regst);
    }
  }
}

void NormalMdUpdtCompTaskNode::ConsumeAllRegsts() {
  if (!IsTrainable()) { return; }
  for (TaskEdge* edge : in_edges()) {
    auto regst_descs = edge->GetRegsts();
    for (auto& regst_desc : regst_descs) {
      ConsumeRegst("model_diff_acc_" + NewUniqueId(), regst_desc);
    }
  }
}

bool NormalMdUpdtCompTaskNode::IsReadyForBuild() {
  return GetProducedRegst("model")->IsLocked() && GetProducedRegst("const_model")->IsLocked();
}

void NormalMdUpdtCompTaskNode::BuildExecGphAndRegst() {
  if (!IsTrainable()) { return; }
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  size_t ibn_idx = 0;
  for (const auto& pair : consumed_regsts()) {
    node->BindBnWithRegst(node->op()->input_bns().Get(ibn_idx++), pair.second.front());
  }
  node->BindBnWithRegst(node->op()->SoleObn(), GetProducedRegst("model"));
  node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, GetProducedRegst("data_tmp"));
  node->AddBnToRegstAndBindIt(&Operator::moving_model_bns, GetProducedRegst("moving_model"));
  node->InferBlobDescs(nullptr);
}

void NormalMdUpdtCompTaskNode::LockRegsts() {
  GetProducedRegst("data_tmp")->Lock();
  GetProducedRegst("moving_model")->Lock();
}

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
