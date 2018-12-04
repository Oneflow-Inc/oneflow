#include "oneflow/core/graph/reduce_inplace_identity_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/reduce_inplace_identity_op.h"

namespace oneflow {

void ReduceInplaceIdentityCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false);
  BindEdgeWithProducedRegst(SoleOutEdge(), "out");
}

void ReduceInplaceIdentityCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void ReduceInplaceIdentityCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnWithRegst(op->SoleIbn(), GetSoleConsumedRegst("in"));

  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(op->BnInOp2Lbi(op->SoleObn()));
  exec_node->BindBnWithRegst(op->SoleObn(), out_regst);
  exec_node->InferBlobDescs(parallel_ctx());
}

void ReduceInplaceIdentityCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
