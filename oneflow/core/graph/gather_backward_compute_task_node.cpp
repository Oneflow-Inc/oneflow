#include "oneflow/core/graph/gather_backward_compute_task_node.h"

namespace oneflow {

void GatherBackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  SoleOutEdge()->AddRegst("in_diff", ProducedRegst("in_diff"));
}

void GatherBackwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    if (IsForwardTaskType(src_node->GetTaskType())) {
      for (TaskEdge* fw_node_in_edge : src_node->in_edges()) {
        TaskNode* pre_fw_node = fw_node_in_edge->src_node();
        if (pre_fw_node->GetTaskType() != TaskType::kMdUpdt) {
          ConsumeRegst("in", edge->GetSoleRegst());
          break;
        }
      }
      UNEXPECTED_RUN();
    } else {
      ConsumeRegst("out_diff", SoleInEdge()->GetSoleRegst());
    }
  }
}

void GatherBackwardCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  node->BindBnInOpAndRegst(node->op()->SoleOdbn(), GetConsumedRegst("out_diff"));
  node->BindBnInOpAndRegst(node->op()->SoleIdbn(), in_diff_regst);

  in_diff_regst->CopyBlobDescWithoutAddLbn(GetConsumedRegst("in"));
}

}
