#include "oneflow/core/graph/nccl_reduce_scatter_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NcclReduceScatterCompTaskNode::EnableMemSharingInReduce(
    std::function<void(RegstDesc* regst, int64_t offset)> EnableMemSharing4Regst) {
  int64_t rank = logical_node()->parallel_desc()->DeviceRank4ParallelId(parallel_id());
  RegstDesc* out_regst = GetProducedRegst("out").get();
  EnableMemSharing4Regst(out_regst, InferRegstSize(*out_regst) * rank);
}

}  // namespace oneflow