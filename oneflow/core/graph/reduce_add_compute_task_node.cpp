#include "oneflow/core/graph/reduce_add_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceAddCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out");
  for (TaskEdge* edge : out_edges()) { BindEdgeWithProducedRegst(edge, "out"); }
}

void ReduceAddCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    while (src_node->GetTaskType() != TaskType::kReduceScatter) {
      src_node = src_node->SoleInEdge()->src_node();
    }
    CompTaskNode* reduce_scatter_node = static_cast<CompTaskNode*>(src_node);
    std::string in_regst_name = "in_" + std::to_string(reduce_scatter_node->parallel_id());
    ConsumeRegst(in_regst_name, edge->GetSoleRegst());
  }
}

void ReduceAddCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> reduce_add_op = this->logical_node()->SoleOp();
  node->mut_op() = reduce_add_op;
  FOR_RANGE(size_t, i, 0, reduce_add_op->input_bns().size()) {
    std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in_" + std::to_string(i));
    node->BindBnWithRegst(reduce_add_op->input_bns().Get(i), in_regst);
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(reduce_add_op->BnInOp2Lbi(reduce_add_op->SoleObn()));
  node->BindBnWithRegst(reduce_add_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
