#include "oneflow/core/graph/reduce_identity_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/reduce_identity_op.h"

namespace oneflow {

void ReduceIdentityCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst_desc = ProduceRegst("out", false);
  this->ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst_desc); });
}

void ReduceIdentityCompTaskNode::EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) {
  ctx.EnableMemSharing4Regst(GetProducedRegst("out").get(), 0);
  ctx.EnableMemSharing4Regst(GetSoleConsumedRegst("in").get(), 0);
}

void ReduceIdentityCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void ReduceIdentityCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnWithRegst(op->SoleIbn(), GetSoleConsumedRegst("in"));

  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(op->BnInOp2Lbi(op->SoleObn()));
  exec_node->BindBnWithRegst(op->SoleObn(), out_regst);
  exec_node->InferBlobDescs(parallel_ctx());
}

void ReduceIdentityCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
