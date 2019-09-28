#include "oneflow/core/graph/nccl_hierarchical_reduce_compute_task_node.h"

namespace oneflow {

void NcclHierarchicalReduceCompTaskNode::EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) {
  ctx.EnableMemSharing4Regst(GetSoleConsumedRegst("in").get(), 0);
}

}  // namespace oneflow
