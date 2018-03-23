#include "oneflow/core/graph/normalization_model_update_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"
namespace oneflow {

void NormalizationMdUpdtCompTaskNode::ProduceAllRegstsAndBindEdges() {
  int32_t min_model_regst = -1;
  int32_t max_model_regst = -1;
  if (JobDesc::Singleton()->IsPredict()) {
    min_model_regst = 1;
    max_model_regst = 1;
  } else if (JobDesc::Singleton()->Staleness() == -1) {
    min_model_regst = 2;
    max_model_regst = kMaxRegisterNum;
  } else {
    min_model_regst = 1;
    max_model_regst = JobDesc::Singleton()->Staleness() + 1;
  }
  std::shared_ptr<RegstDesc> norm_model_regst =
      ProduceRegst("norm_model", min_model_regst, max_model_regst);
  for (TaskEdge* out_edge : out_edges()) {
    out_edge->AddRegst("norm_model", norm_model_regst);
  }
}

void NormalizationMdUpdtCompTaskNode::ConsumeAllRegsts() {
  if (JobDesc::Singleton()->IsPredict()) { return; }
  for (TaskEdge* in_edge : in_edges()) {
    ConsumeRegst("norm_acc", in_edge->GetSoleRegst());
  }
}

bool NormalizationMdUpdtCompTaskNode::IsReadyForBuild() {
  return GetProducedRegst("norm_model")->IsLocked();
}

void NormalizationMdUpdtCompTaskNode::BuildExecGphAndRegst() {
  if (JobDesc::Singleton()->IsPredict()) { return; }
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  for (const std::string& obn : node->op()->output_bns()) {
    node->BindBnInOpAndRegst(obn, GetProducedRegst("norm_model"));
  }
  if (JobDesc::Singleton()->IsTrain()) {
    for (const std::string& ibn : node->op()->input_bns()) {
      node->BindBnInOpAndRegst(ibn, GetConsumedRegst("norm_acc"));
    }
  }
}

void NormalizationMdUpdtCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  ForEachNodeOnOutEdge([&](const TaskNode* node) {
    if (IsForwardTaskType(node->GetTaskType())) {
      if (task_proto->related_init_model_task_id() == -1) {
        task_proto->set_related_init_model_task_id(node->task_id());
      }
    } else {
      CHECK_EQ(task_proto->related_save_model_task_id(), -1);
      task_proto->set_related_save_model_task_id(node->task_id());
    }
  });
}

}  // namespace oneflow
