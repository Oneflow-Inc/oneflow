#include "oneflow/core/graph/reduce_global_add_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceGlobalAddCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out");
  for (TaskEdge* edge : out_edges()) { BindEdgeWithProducedRegst(edge, "out"); }
  ProduceRegst("data_tmp", 1, 1);
}

void ReduceGlobalAddCompTaskNode::ConsumeAllRegsts() {
  int32_t in_regst_idx = 0;
  for (TaskEdge* edge : in_edges()) {
    ConsumeRegst("in_" + std::to_string(in_regst_idx), edge->GetSoleRegst());
    ++in_regst_idx;
  }
}

void ReduceGlobalAddCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> reduce_global_add_op = this->logical_node()->SoleOp();
  node->mut_op() = reduce_global_add_op;
  FOR_RANGE(size_t, i, 0, reduce_global_add_op->input_bns().size()) {
    std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in_" + std::to_string(i));
    node->BindBnWithRegst(reduce_global_add_op->input_bns().Get(i), in_regst);
  }
  node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, GetProducedRegst("data_tmp"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(reduce_global_add_op->BnInOp2Lbi(reduce_global_add_op->SoleObn()));
  node->BindBnWithRegst(reduce_global_add_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
