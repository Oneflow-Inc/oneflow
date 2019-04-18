#include "oneflow/core/graph/nccl_all_reduce_compute_task_node.h"

namespace oneflow {

void NcclAllReduceCompTaskNode::EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) {
  int64_t offset = ctx.Offset4RankCtxParallelId(GetRankCtx().CtxWithGather(), parallel_id());
  ctx.EnableMemSharing4Regst(GetProducedRegst("out").get(), offset);
  if (this->SoleInDataEdge()->src_node()->GetTaskType() == TaskType::kReduceConcat) { return; }

  ctx.EnableMemSharing4Regst(GetSoleConsumedRegst("in").get(), offset);
}

}  // namespace oneflow
