#ifndef ONEFLOW_CORE_GRAPH_NCCL_HIERARCHICAL_BROADCAST_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NCCL_HIERARCHICAL_BROADCAST_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/pipe_compute_task_node.h"
#include "oneflow/core/graph/reduce_comp_task_node_if.h"

namespace oneflow {

class NcclHierarchicalBroadcastCompTaskNode final : public PipeCompTaskNode,
                                                    public ReduceCompTaskNodeIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclHierarchicalBroadcastCompTaskNode);
  NcclHierarchicalBroadcastCompTaskNode() = default;
  ~NcclHierarchicalBroadcastCompTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kNcclHierarchicalReduce; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kNccl; }

  void EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NCCL_HIERARCHICAL_BROADCAST_COMPUTE_TASK_NODE_H_
