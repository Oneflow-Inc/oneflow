#ifndef ONEFLOW_CORE_GRAPH_NCCL_REDUCE_SCATTER_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NCCL_REDUCE_SCATTER_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/poor_compute_task_node.h"

namespace oneflow {

class NcclReduceScatterCompTaskNode final : public PoorCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclReduceScatterCompTaskNode);
  NcclReduceScatterCompTaskNode() = default;
  ~NcclReduceScatterCompTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kNcclReduceScatter; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kNcclScatter; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NCCL_REDUCE_SCATTER_COMPUTE_TASK_NODE_H_
