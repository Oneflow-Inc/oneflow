#include "oneflow/core/graph/repeat_backward_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void RepeatBackwardCompTaskNode::ConsumeAllRegsts() {
  for (const TaskEdge* edge : in_edges()) {
    if (edge->src_node()->GetTaskType() == TaskType::kRepeatForward) { continue; }
    CHECK(consumed_regsts().empty());
    ConsumeRegst("out_diff", edge->GetSoleRegst());
  }
}

void RepeatBackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("in_diff", false, 1, 1);
  for (TaskEdge* edge : out_edges()) { edge->AddRegst("in_diff", out_regst); }
}

void RepeatBackwardCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> sole_op = this->logical_node()->SoleOp();
  node->mut_op() = sole_op;
  node->BindBnWithRegst(sole_op->SoleOdbn(), GetSoleConsumedRegst("out_diff"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("in_diff");
  out_regst->AddLbi(sole_op->BnInOp2Lbi(sole_op->SoleIdbn()));
  node->BindBnWithRegst(sole_op->SoleIdbn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void RepeatBackwardCompTaskNode::InferProducedDataRegstTimeShape() {
  std::vector<int64_t> time_shape_dim_vec =
      GetSoleConsumedRegst("out_diff")->data_regst_time_shape()->dim_vec();
  CHECK(this->logical_node()->SoleOp()->op_conf().has_repeat_conf());
  int64_t repeat_num = this->logical_node()->SoleOp()->op_conf().repeat_conf().repeat_num();
  CHECK(!time_shape_dim_vec.empty());
  CHECK(time_shape_dim_vec.back() == repeat_num);
  time_shape_dim_vec.pop_back();
  std::shared_ptr<Shape> time_shape(new Shape(time_shape_dim_vec));
  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

}  // namespace oneflow
