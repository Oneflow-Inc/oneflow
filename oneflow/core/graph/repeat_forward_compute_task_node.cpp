#include "oneflow/core/graph/repeat_forward_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/repeat_op.h"

namespace oneflow {

void RepeatForwardCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void RepeatForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst); });
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
  DimVector time_shape_dim_vec = GetSoleConsumedRegst("in")->data_regst_time_shape()->dim_vec();
  const RepeatOp* repeat_op = dynamic_cast<RepeatOp*>(this->logical_node()->SoleOp().get());
  CHECK_NOTNULL(repeat_op);
  int32_t repeat_num = repeat_op->GetRepeatNum();
  time_shape_dim_vec.push_back(repeat_num);
  GetProducedRegst("out")->mut_data_regst_time_shape()->reset(new Shape(time_shape_dim_vec));
}

}  // namespace oneflow
