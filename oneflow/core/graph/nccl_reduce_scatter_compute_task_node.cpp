#include "oneflow/core/graph/nccl_reduce_scatter_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NcclReduceScatterCompTaskNode::EnableMemSharingInReduce(ReduceMemSharingCtx* ctx) {
  int64_t offset = ctx->Offset4ParallelId(parallel_id());
  int64_t rank = ctx->StageRank4ParallelId(parallel_id());
  RegstDesc* out_regst = GetProducedRegst("out").get();
  int64_t out_size = InferRegstSize(*out_regst);
  CHECK_EQ(ctx->StageSegmentSize() % out_size, 0);
  ctx->EnableMemSharing4Regst(out_regst, offset + out_size * rank);
  ctx->DoScatter(ctx->StageSegmentSize() / out_size);
  if (this->SoleInEdge()->src_node()->GetTaskType() == TaskType::kReduceConcat) { return; }

  ctx->EnableMemSharing4Regst(GetSoleConsumedRegst("in").get(), offset);
}

}  // namespace oneflow
