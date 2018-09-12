#include "oneflow/core/graph/normal_model_update_compute_task_node.h"
#include "oneflow/core/graph/reduce_split_compute_task_node.h"
#include "oneflow/core/graph/normal_backward_compute_task_node.h"
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

CompTaskNode* NormalMdUpdtCompTaskNode::FindReduceSplitCompTaskNode() {
  for (TaskEdge* edge : this->in_edges()) {
    CompTaskNode* comp_task_node = dynamic_cast<CompTaskNode*>(edge->src_node());
    CHECK(comp_task_node != nullptr);
    if (comp_task_node->GetTaskType() == TaskType::kReduceSplit) { return comp_task_node; }
  }
  return nullptr;
}

CompTaskNode* NormalMdUpdtCompTaskNode::FindCorrespondingBackwardCompTaskNode() {
  CompTaskNode* cur_node = this;
  while (cur_node->GetTaskType() != TaskType::kNormalBackward) {
    for (TaskEdge* edge : cur_node->in_edges()) {
      CompTaskNode* comp_task_node = dynamic_cast<CompTaskNode*>(edge->src_node());
      if (comp_task_node != nullptr) { cur_node = comp_task_node; }
    }
  }
  return cur_node;
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
  node->InferBlobDescs(nullptr);

  // set instance num
  bool has_instance_num = false;
  CompTaskNode* split_node = FindReduceSplitCompTaskNode();
  if (split_node) {
    std::shared_ptr<RegstDesc> out_regst = split_node->GetProducedRegst("out_0");
    out_regst->ForEachLbi([&](const LogicalBlobId& lbi) {
      if (out_regst->GetBlobDesc(lbi)->has_instance_num_field()) { has_instance_num = true; }
    });
  } else {
    CompTaskNode* bw_node = FindCorrespondingBackwardCompTaskNode();
    std::shared_ptr<RegstDesc> in_diff_regst = bw_node->GetProducedRegst("in_diff");
    in_diff_regst->ForEachLbi([&](const LogicalBlobId& lbi) {
      if (in_diff_regst->GetBlobDesc(lbi)->has_instance_num_field()) { has_instance_num = true; }
    });
  }
  if (has_instance_num) {
    std::shared_ptr<RegstDesc> model_regst = GetProducedRegst("model");
    model_regst->ForEachLbi([&](const LogicalBlobId& lbi) {
      model_regst->MutBlobDesc(lbi)->set_has_instance_num_field(true);
    });
  }
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
