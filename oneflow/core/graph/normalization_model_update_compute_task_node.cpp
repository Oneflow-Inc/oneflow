#include "oneflow/core/graph/normalization_model_update_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"
namespace oneflow {

void NormalizationMdUpdtCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> other_model_regst =
      ProduceRegst("other_model", 1, 1);
  for (TaskEdge* out_edge : out_edges()) {
    out_edge->AddRegst("other_model", other_model_regst);
  }
}

bool NormalizationMdUpdtCompTaskNode::IsReadyForBuild() {
  return GetProducedRegst("other_model")->IsLocked();
}

void NormalizationMdUpdtCompTaskNode::BuildExecGphAndRegst() {
  if (JobDesc::Singleton()->IsPredict()) { return; }
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf op_conf;
  op_conf.set_name("norm_md_update_" + NewUniqueId());
  op_conf.mutable_normalization_mdupdt_conf();
  node->mut_op() = ConstructOp(op_conf);
  node->BindBnInOpAndRegst(node->op()->SoleObn(),
                           GetProducedRegst("other_model"));
}

void NormalizationMdUpdtCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  ForEachNodeOnOutEdge([&](const TaskNode* node) {
    if (IsForwardTaskType(node->GetTaskType())) {
      if (task_proto->related_init_model_task_id() == -1) {
        task_proto->set_related_init_model_task_id(node->task_id());
      }
    } else if (IsBackwardTaskType(node->GetTaskType())) {
      // do nothing
    } else if (node->GetTaskType() == TaskType::kMdSave) {
      CHECK_EQ(task_proto->related_save_model_task_id(), -1);
      task_proto->set_related_save_model_task_id(node->task_id());
    } else {
      UNIMPLEMENTED();
    }
  });
}

}  // namespace oneflow
