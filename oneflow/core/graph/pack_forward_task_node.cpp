#include "oneflow/core/graph/pack_forward_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void PackForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false);
  for (TaskEdge* edge : out_edges()) {
    const LogicalNode* succ_logical = GetOneSuccLogicalNodeOnEdge(edge);
    if (succ_logical->TypeName() != "PackBackward") { BindEdgeWithProducedRegst(edge, "out"); }
  }
}

void PackForwardCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void PackForwardCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnWithRegst(op->SoleIbn(), GetSoleConsumedRegst("in"));

  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(op->BnInOp2Lbi(op->SoleObn()));
  exec_node->BindBnWithRegst(op->SoleObn(), out_regst);
  exec_node->InferBlobDescs(parallel_ctx());
}

void PackForwardCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::vector<int64_t> time_shape_dim_vec(in_regst->data_regst_time_shape()->dim_vec());

  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  int64_t in_piece_size = in_regst->GetBlobDesc(op->BnInOp2Lbi("in"))->shape().At(0);
  int64_t out_piece_size = op->op_conf().pack_conf().out_size();
  CHECK_GT(time_shape_dim_vec.size(), 0);
  CHECK_EQ(time_shape_dim_vec.back(), RoundUp(out_piece_size, in_piece_size));
  time_shape_dim_vec.pop_back();
  *out_regst->mut_data_regst_time_shape() = std::make_shared<Shape>(std::move(time_shape_dim_vec));
}

}  // namespace oneflow
