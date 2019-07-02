#include "oneflow/core/graph/instance_stack_forward_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/instance_stack_op.h"

namespace oneflow {

void InstanceStackForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false);
  for (TaskEdge* edge : out_edges()) {
    const LogicalNode* succ_logical = GetOneSuccLogicalNodeOnEdge(edge);
    if (succ_logical->TypeName() != "InstanceStackBackward") {
      BindEdgeWithProducedRegst(edge, "out");
    }
  }
}

void InstanceStackForwardCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void InstanceStackForwardCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  exec_node->BindBnWithRegst(op->SoleIbn(), in_regst);
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  const LogicalBlobId out_lbi = op->BnInOp2Lbi(op->SoleObn());
  out_regst->AddLbi(out_lbi);
  exec_node->BindBnWithRegst(op->SoleObn(), out_regst);

  CHECK_EQ(related_piece_slice_->consumed_regsts().size(), 1);
  CHECK_EQ(related_piece_slice_->consumed_regsts().begin()->second.size(), 1);
  std::shared_ptr<RegstDesc> related_piece_slice_consumed_regst =
      (*related_piece_slice_->consumed_regsts().begin()).second.front();
  *(out_regst->MutSoleBlobDesc()) = *(related_piece_slice_consumed_regst->SoleBlobDesc());
}

void InstanceStackForwardCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  std::vector<int64_t> dim_vec(in_regst->data_regst_time_shape()->dim_vec());
  dim_vec.pop_back();
  CHECK_EQ(related_piece_slice_->consumed_regsts().size(), 1);
  CHECK_EQ(related_piece_slice_->consumed_regsts().begin()->second.size(), 1);
  std::shared_ptr<RegstDesc> related_piece_slice_consumed_regst =
      (*related_piece_slice_->consumed_regsts().begin()).second.front();
  std::shared_ptr<Shape> out_time_shape = std::make_shared<Shape>(dim_vec);
  CHECK_EQ(out_time_shape, related_piece_slice_consumed_regst->data_regst_time_shape());
  *(GetProducedRegst("out")->mut_data_regst_time_shape()) = out_time_shape;
}

}  // namespace oneflow
