#include "oneflow/core/graph/gather_backward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void GatherBackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  SoleOutEdge()->AddRegst("in_diff", ProduceRegst("in_diff"));
}

void GatherBackwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    if (IsForwardTaskType(src_node->GetTaskType())) {
      ConsumeRegst("in", src_node->SoleInEdge()->GetSoleRegst());
    } else {
      ConsumeRegst("out_diff", edge->GetSoleRegst());
    }
  }
}

void GatherBackwardCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  node->BindBnInOpAndRegst(node->op()->SoleIbn(), GetConsumedRegst("in"));
  node->BindBnInOpAndRegst(node->op()->SoleOdbn(),
                           GetConsumedRegst("out_diff"));
  node->BindBnInOpAndRegst(node->op()->SoleIdbn(), in_diff_regst);

  in_diff_regst->AddLbn(node->op()->Lbn4BnInOp(node->op()->SoleIdbn()));

  in_diff_regst->CopyBlobDescWithoutAddLbn(GetConsumedRegst("in").get());
}

}  // namespace oneflow
