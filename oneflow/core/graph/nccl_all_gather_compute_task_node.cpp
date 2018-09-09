#include "oneflow/core/graph/nccl_all_gather_compute_task_node.h"

namespace oneflow {

void NcclAllGatherCompTaskNode::EnableMemSharingInReduce(ReduceMemSharingCtx *ctx) {
  ctx->Gather(ctx->LastCount());
  ctx->EnableMemSharing4Regst(GetProducedRegst("out").get(), ctx->Offset4ParallelId(parallel_id()));
}

}  // namespace oneflow
