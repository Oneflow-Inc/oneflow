#include "oneflow/core/graph/piece_slice_forward_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/piece_slice_op.h"

namespace oneflow {

void PieceSliceForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false);
  for (TaskEdge* edge : out_edges()) {
    const LogicalNode* succ_logical = GetOneSuccLogicalNodeOnEdge(edge);
    if (succ_logical->TypeName() != "PieceSliceBackward") {
      BindEdgeWithProducedRegst(edge, "out");
    }
  }
}

void PieceSliceForwardCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void PieceSliceForwardCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnWithRegst(op->SoleIbn(), GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(op->BnInOp2Lbi(op->SoleObn()));
  exec_node->BindBnWithRegst(op->SoleObn(), out_regst);
  exec_node->InferBlobDescs(parallel_ctx());
}

void PieceSliceForwardCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  std::vector<int64_t> dim_vec(in_regst->data_regst_time_shape()->dim_vec());
  const PieceSliceOp* op = dynamic_cast<PieceSliceOp*>(logical_node()->SoleOp().get());
  CHECK_NOTNULL(op);
  dim_vec.push_back(in_regst->GetBlobDesc(op->BnInOp2Lbi("in"))->shape().At(0));
  *(GetProducedRegst("out")->mut_data_regst_time_shape()) = std::make_shared<Shape>(dim_vec);
}

}  // namespace oneflow
