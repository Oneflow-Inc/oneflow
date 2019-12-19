#ifndef ONEFLOW_CORE_GRAPH_NCCL_TUPLE_BROADCAST_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NCCL_TUPLE_BROADCAST_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class NcclTupleBroadcastCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclTupleBroadcastCompTaskNode);
  NcclTupleBroadcastCompTaskNode() = default;
  ~NcclTupleBroadcastCompTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kNcclTupleBroadcast; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kMix; }

 private:
  void BuildExecGphAndRegst() override;
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() final;
  void InferProducedDataRegstTimeShape() final;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NCCL_TUPLE_BROADCAST_COMPUTE_TASK_NODE_H_
