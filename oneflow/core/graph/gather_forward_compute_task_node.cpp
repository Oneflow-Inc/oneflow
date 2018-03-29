#include "oneflow/core/graph/gather_forward_compute_task_node.h"

namespace oneflow {

void GatherForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  for (TaskEdge* edge : out_edges()) {
    if (IsBackwardTaskType(edge->dst_node()->GetTaskType()) == false) {
      edge->AddRegst("out", ProduceRegst("out"));
    }
  }
}

void GatherForwardCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void GatherForwardCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  node->BindBnInOpAndRegst(node->op()->SoleIbn(), GetConsumedRegst("in"));
  node->BindBnInOpAndRegst(node->op()->SoleObn(), GetProducedRegst("out"));
  CHECK(node->op()->data_tmp_bns().empty());

  node->op()->InferBlobDescs(node->GetBlobDesc4BnInOpFunc(), parallel_ctx(),
                             device_type());
}

} // namespace oneflow
