#include "oneflow/core/graph/nccl_tuple_reduce_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NcclTupleReduceCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out = ProduceRegst("out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* out_edge) { out_edge->AddRegst("out", out); });
}

void NcclTupleReduceCompTaskNode::ConsumeAllRegsts() {
  ForEachInDataEdge([&](TaskEdge* in_edge) { ConsumeRegst("in", in_edge->GetSoleRegst()); });
}

void NcclTupleReduceCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> sole_op = this->logical_node()->SoleOp();
  node->mut_op() = sole_op;
  for (const std::string& ibn : sole_op->input_bns()) {
    node->BindBnWithOneOfTheRegsts(ibn, GetConsumedRegst("in"));
  }
  const NcclTupleReduceOpConf& conf = sole_op->op_conf().nccl_tuple_reduce_conf();
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  FOR_RANGE(int64_t, i, 0, sole_op->output_bns().size()) {
    if (conf.root(i) == parallel_id()) {
      const std::string& obn = sole_op->output_bns().Get(i);
      out_regst->AddLbi(sole_op->BnInOp2Lbi(obn));
      node->BindBnWithRegst(obn, out_regst);
    }
  }
  node->InferBlobDescs(parallel_ctx());
}

void NcclTupleReduceCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
