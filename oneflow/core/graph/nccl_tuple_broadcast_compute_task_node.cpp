#include "oneflow/core/graph/nccl_tuple_broadcast_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NcclTupleBroadcastCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out = ProduceRegst("out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* out_edge) { out_edge->AddRegst("out", out); });
}

void NcclTupleBroadcastCompTaskNode::ConsumeAllRegsts() {
  const Operator* op = logical_node()->SoleOp().get();
  HashMap<LogicalBlobId, int64_t> lbi2ibn_id;
  FOR_RANGE(int64_t, i, 0, op->output_bns().size()) {
    lbi2ibn_id.emplace(op->BnInOp2Lbi(GenRepeatedBn("in", i)), i);
  }
  ForEachInDataEdge([&](TaskEdge* out_edge) {
    const LogicalNode* pred = GetOnePredLogicalNodeOnEdge(out_edge);
    CHECK_NOTNULL(pred);
    const Operator* pred_op = pred->SoleOp().get();
    auto it = lbi2ibn_id.find(pred_op->BnInOp2Lbi(pred_op->SoleObn()));
    if (it != lbi2ibn_id.end()) {
      ConsumeRegst("in_" + std::to_string(it->second), out_edge->GetSoleRegst());
    } else {
      ConsumeRegst("in", out_edge->GetSoleRegst());
    }
  });
}

void NcclTupleBroadcastCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> sole_op = this->logical_node()->SoleOp();
  node->mut_op() = sole_op;
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  ForEachConsumedDataRegst([&](const std::string& name, const RegstDesc* regst) {
    node->BindBnWithRegst(name, GetSoleConsumedRegst(name));
  });
  for (const std::string& obn : sole_op->output_bns()) {
    out_regst->AddLbi(sole_op->BnInOp2Lbi(obn));
    node->BindBnWithRegst(obn, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void NcclTupleBroadcastCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
