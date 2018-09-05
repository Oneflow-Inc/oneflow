#ifndef ONEFLOW_CORE_GRAPH_NCCL_ALL_REDUCE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NCCL_ALL_REDUCE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/poor_compute_task_node.h"

namespace oneflow {

class NcclAllReduceCompTaskNode final : public PoorCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclAllReduceCompTaskNode);
  NcclAllReduceCompTaskNode() = default;
  ~NcclAllReduceCompTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kNcclAllReduce; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kMix; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NCCL_ALL_REDUCE_COMPUTE_TASK_NODE_H_
