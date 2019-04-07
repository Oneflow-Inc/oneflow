#include "oneflow/core/graph/keep_header_only_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/keep_header_only_op.h"

namespace oneflow {

void KeepHeaderOnlyCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false, 100, 100);
  BindEdgeWithProducedRegst(SoleOutEdge(), "out");
}

void KeepHeaderOnlyCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void KeepHeaderOnlyCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnsWithRegst(&Operator::input_bns, GetSoleConsumedRegst("in"));

  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  for (const std::string& obn : op->output_bns()) { out_regst->AddLbi(op->BnInOp2Lbi(obn)); }
  exec_node->BindBnsWithRegst(&Operator::output_bns, out_regst);
  exec_node->InferBlobDescs(parallel_ctx());
}

void KeepHeaderOnlyCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
