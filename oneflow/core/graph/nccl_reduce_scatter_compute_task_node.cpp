#include "oneflow/core/graph/nccl_reduce_scatter_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NcclReduceScatterCompTaskNode::EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) {
  const ReduceRankingCtx& ranking_ctx = GetRankingCtx();
  int64_t offset = ctx.Offset4RankingParallelId(ranking_ctx, parallel_id());
  RegstDesc* out_regst = GetProducedRegst("out").get();
  ctx.EnableMemSharing4Regst(out_regst, offset);
  if (this->SoleInEdge()->src_node()->GetTaskType() == TaskType::kReduceConcat) { return; }
  ctx.EnableMemSharing4Regst(
      GetSoleConsumedRegst("in").get(),
      ctx.Offset4RankingParallelId(ranking_ctx.CtxWithGather(), parallel_id()));
}

}  // namespace oneflow
