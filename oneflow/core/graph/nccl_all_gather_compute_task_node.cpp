#include "oneflow/core/graph/nccl_all_gather_compute_task_node.h"

namespace oneflow {

void NcclAllGatherCompTaskNode::EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) {
  ctx.EnableMemSharing4Regst(
      GetProducedRegst("out").get(),
      ctx.Offset4RankCtxParallelId(GetRankCtx().CtxWithGather(), parallel_id()));
}

}  // namespace oneflow
