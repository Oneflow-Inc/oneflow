#include "oneflow/core/graph/nccl_all_reduce_compute_task_node.h"

namespace oneflow {

void NcclAllReduceCompTaskNode::EnableMemSharingInReduce(
    std::function<void(RegstDesc* regst, int64_t offset)> EnableMemSharing4Regst) {
  EnableMemSharing4Regst(GetProducedRegst("out").get(), 0);
  if (this->SoleInEdge()->src_node()->GetTaskType() == TaskType::kReduceConcat) { return; }

  EnableMemSharing4Regst(GetSoleConsumedRegst("in").get(), 0);
}

}  // namespace oneflow