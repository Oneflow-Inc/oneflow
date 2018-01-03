#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/recurrent_forward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void RecurrentForwardCompTaskNode::VirtualConsumeInRegst(TaskEdge* edge) {
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  std::shared_ptr<RegstDesc> regst = edge->GetSoleRegst();
  if (regst->GetBlobDesc(op->Lbn4BnInOp("in"))) {
    ConsumeRegst("in", regst);
  } else if (regst->GetBlobDesc(op->Lbn4BnInOp("h0"))) {
    ConsumeRegst("h0", regst);
  } else if (regst->GetBlobDesc(op->Lbn4BnInOp("ht_1"))) {
    ConsumeRegst("ht_1", regst);
  } else {
    UNEXPECTED_RUN();
  }
}

}  // namespace oneflow
