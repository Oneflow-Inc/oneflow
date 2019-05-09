#ifndef ONEFLOW_CORE_GRAPH_LOCAL_GPU_PEER_BOXING_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_LOCAL_GPU_PEER_BOXING_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class LocalGpuPeerBoxingCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalGpuPeerBoxingCompTaskNode);
  LocalGpuPeerBoxingCompTaskNode() = default;
  ~LocalGpuPeerBoxingCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kLocalGpuPeerBoxing; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kMix; }

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOCAL_GPU_PEER_CONCAT_SPLIT_BOXING_COMPUTE_TASK_NODE_H_
