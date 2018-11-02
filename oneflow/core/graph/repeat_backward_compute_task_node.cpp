#include "oneflow/core/graph/repeat_backward_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/repeat_op.h"

namespace oneflow {

void RepeatBackwardCompTaskNode::ConsumeAllRegsts() {
  for (const TaskEdge* edge : in_edges()) {
    if (edge->src_node()->GetTaskType() == TaskType::kRepeatForward) { continue; }
    CHECK(consumed_regsts().empty());
    ConsumeRegst("out_diff", edge->GetSoleRegst());
  }
}

void RepeatBackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> in_diff_regst = ProduceRegst("in_diff", false, 1, 1);
  SoleOutEdge()->AddRegst("in_diff", in_diff_regst);
}

void RepeatBackwardCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> sole_op = this->logical_node()->SoleOp();
  node->mut_op() = sole_op;
  node->BindBnWithRegst(sole_op->SoleOdbn(), GetSoleConsumedRegst("out_diff"));
  std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff");
  in_diff_regst->AddLbi(sole_op->BnInOp2Lbi(sole_op->SoleIdbn()));
  node->BindBnWithRegst(sole_op->SoleIdbn(), in_diff_regst);
  node->InferDiffBlobDescsWithoutFwNode(parallel_ctx());
}

void RepeatBackwardCompTaskNode::InferProducedDataRegstTimeShape() {
  std::vector<int64_t> time_shape_dim_vec =
      GetSoleConsumedRegst("out_diff")->data_regst_time_shape()->dim_vec();
  CHECK(this->logical_node()->SoleOp()->op_conf().has_repeat_conf());
  const RepeatOp* repeat_op = dynamic_cast<RepeatOp*>(this->logical_node()->SoleOp().get());
  CHECK_NOTNULL(repeat_op);
  int32_t repeat_num = repeat_op->GetRepeatNum(parallel_ctx()->parallel_num());
  CHECK(!time_shape_dim_vec.empty());
  CHECK(time_shape_dim_vec.back() == repeat_num);
  time_shape_dim_vec.pop_back();
  GetProducedRegst("in_diff")->mut_data_regst_time_shape()->reset(new Shape(time_shape_dim_vec));
}

}  // namespace oneflow
