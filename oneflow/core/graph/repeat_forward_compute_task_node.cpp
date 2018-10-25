#include "oneflow/core/graph/repeat_forward_compute_task_node.h"
#include "oneflow/core/graph/repeat_backward_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void RepeatForwardCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void RepeatForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 1, 1);
  for (TaskEdge* edge : out_edges()) {
    if (edge->dst_node()->GetTaskType() == TaskType::kRepeatBackward) { continue; }
    edge->AddRegst("out", out_regst);
  }
}

void RepeatForwardCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> sole_op = this->logical_node()->SoleOp();
  node->mut_op() = sole_op;
  node->BindBnWithRegst(sole_op->SoleIbn(), GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(sole_op->BnInOp2Lbi(sole_op->SoleObn()));
  node->BindBnWithRegst(sole_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void RepeatForwardCompTaskNode::InferProducedDataRegstTimeShape() {
  std::vector<int64_t> time_shape_dim_vec =
      GetSoleConsumedRegst("in")->data_regst_time_shape()->dim_vec();
  CHECK(this->logical_node()->SoleOp()->op_conf().has_repeat_conf());
  time_shape_dim_vec.push_back(
      this->logical_node()->SoleOp()->op_conf().repeat_conf().repeat_num());
  GetProducedRegst("out")->mut_data_regst_time_shape()->reset(new Shape(time_shape_dim_vec));
}

}  // namespace oneflow
