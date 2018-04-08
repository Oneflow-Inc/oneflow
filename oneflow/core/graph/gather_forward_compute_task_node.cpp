#include "oneflow/core/graph/gather_forward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

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
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");

  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  node->BindBnInOpAndRegst(node->op()->SoleIbn(), GetConsumedRegst("in"));
  node->BindBnInOpAndRegst(node->op()->SoleObn(), out_regst);
  CHECK(node->op()->data_tmp_bns().empty());

  out_regst->AddLbn(node->op()->Lbn4BnInOp(node->op()->SoleObn()));

  node->op()->InferBlobDescs(node->GetBlobDesc4BnInOpFunc(), parallel_ctx(),
                             device_type());
}

}  // namespace oneflow
