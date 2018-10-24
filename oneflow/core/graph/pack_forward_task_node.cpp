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

  const auto& related_up_consumed_regsts = related_unpack_->consumed_regsts();
  CHECK_EQ(1, related_up_consumed_regsts.size());
  CHECK_EQ(1, (*related_up_consumed_regsts.begin()).second.size());
  std::shared_ptr<RegstDesc> related_up_consumed_regst =
      (*related_up_consumed_regsts.begin()).second.front();
  *out_regst->MutSoleBlobDesc() = *related_up_consumed_regst->SoleBlobDesc();
}

void PackForwardCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::vector<int64_t> time_shape_dim_vec(in_regst->data_regst_time_shape()->dim_vec());

  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  int64_t pack_num = op->op_conf().pack_conf().pack_num();
  CHECK_GT(time_shape_dim_vec.size(), 0);
  CHECK_EQ(pack_num, time_shape_dim_vec.back());
  time_shape_dim_vec.pop_back();

  std::shared_ptr<RegstDesc> related_up_consumed_regst =
      (*related_unpack_->consumed_regsts().begin()).second.front();
  std::shared_ptr<Shape> pack_out_time_shape =
      std::make_shared<Shape>(std::move(time_shape_dim_vec));
  CHECK_EQ(*pack_out_time_shape, *(related_up_consumed_regst->data_regst_time_shape()));
  *(out_regst->mut_data_regst_time_shape()) = pack_out_time_shape;
}

}  // namespace oneflow
