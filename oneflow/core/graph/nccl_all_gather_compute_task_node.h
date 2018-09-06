#ifndef ONEFLOW_CORE_GRAPH_NCCL_ALL_GATHER_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NCCL_ALL_GATHER_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/poor_compute_task_node.h"

namespace oneflow {

class NcclAllGatherCompTaskNode final : public PoorCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclAllGatherCompTaskNode);
  NcclAllGatherCompTaskNode() = default;
  ~NcclAllGatherCompTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kNcclAllGather; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kNcclGather; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NCCL_ALL_GATHER_COMPUTE_TASK_NODE_H_
