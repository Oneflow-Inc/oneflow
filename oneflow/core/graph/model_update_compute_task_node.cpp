#include "oneflow/core/graph/model_update_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void MdUpdtCompTaskNode::ProduceAllRegstsAndBindEdges() {
  int32_t min_model_regst = 1;
  if (JobDesc::Singleton()->Staleness() == -1) { min_model_regst = 2; }
  auto model_regst = ProduceRegst("model", min_model_regst, kMaxRegisterNum);
  auto model_tmp_regst = ProduceRegst("model_tmp", 1, 1);
  for (TaskEdge* out_edge : out_edges()) {
    TaskNode* dst_node = out_edge->dst_node();
    if (dst_node->GetTaskType() == TaskType::kForward
        || dst_node->GetTaskType() == TaskType::kBackward) {
      out_edge->AddRegst("model", model_regst);
      out_edge->AddRegst("model_tmp", model_tmp_regst);
    } else if (dst_node->GetTaskType() == TaskType::kMdSave) {
      out_edge->AddRegst("model", model_regst);
    } else {
      UNEXPECTED_RUN();
    }
  }
}

void MdUpdtCompTaskNode::ConsumeAllRegsts() {
  if (JobDesc::Singleton()->IsPredict()) { return; }
  ConsumeRegst("model_diff_acc", SoleInEdge()->GetSoleRegst());
}

bool MdUpdtCompTaskNode::IsReadyForBuild() {
  return GetProducedRegst("model")->IsLocked()
         && GetProducedRegst("model_tmp")->IsLocked();
}

void MdUpdtCompTaskNode::BuildExecGphAndRegst() {
  if (JobDesc::Singleton()->IsPredict()) { return; }
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  node->BindBnInOpAndRegst(node->op()->SoleIbn(), SoleInEdge()->GetSoleRegst());
  node->BindBnInOpAndRegst(node->op()->SoleObn(), GetProducedRegst("model"));
  auto data_tmp_regst = ProduceRegst("data_tmp", 1, 1);
  for (const std::string& dtbn : node->op()->data_tmp_bns()) {
    const std::string& lbn = node->op()->Lbn4BnInOp(dtbn);
    data_tmp_regst->AddLbn(lbn);
    node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
  }
  node->op()->InferBlobDescs(node->GetBlobDesc4BnInOpFunc(), nullptr);
}

void MdUpdtCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  task_proto->set_random_seed(random_seed_);
  ForEachNodeOnOutEdge([&](const TaskNode* node) {
    if (node->GetTaskType() == TaskType::kForward) {
      CHECK_EQ(task_proto->related_fw_task_id(), -1);
      task_proto->set_related_fw_task_id(node->task_id());
    } else if (node->GetTaskType() == TaskType::kBackward) {
      // do nothing
    } else {
      CHECK_EQ(task_proto->related_save_task_id(), -1);
      task_proto->set_related_save_task_id(node->task_id());
    }
  });
}

}  // namespace oneflow
