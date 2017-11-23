#include "oneflow/core/graph/loss_accumulate_compute_task_node.h"

namespace oneflow {

void AccCompTaskNode::ProduceAllRegstsAndBindEdges() {
  auto acc_regst = ProduceRegst("acc", 1, kMaxRegisterNum);
  SoleOutEdge()->AddRegst("acc", acc_regst);
}

void AccCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("one", SoleInEdge()->GetSoleRegst());
}

}  // namespace oneflow
