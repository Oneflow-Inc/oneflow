#include "oneflow/core/graph/loss_accumulate_compute_task_node.h"

namespace oneflow {

void AccCompTaskNode::ProduceAllRegstsAndBindEdges() {
  auto acc_regst = ProduceRegst("acc", 1, kMaxRegisterNum);
  for (TaskEdge* edge : out_edges()) { edge->AddRegst("acc", acc_regst); }
}

void AccCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    ConsumeRegst("one", edge->GetSoleRegst());
  }
}

}  // namespace oneflow
