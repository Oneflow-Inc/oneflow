#ifndef ONEFLOW_CORE_GRAPH_CUDA_COPY_PEER_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_CUDA_COPY_PEER_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class CudaCopyPeerTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaCopyPeerTaskNode);
  CudaCopyPeerTaskNode() = default;
  ~CudaCopyPeerTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kCudaCopyPeer; }

  void Init(int64_t machine_id, int64_t thrd_id);
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;
  OperatorConf NewCopyOpConf();

 private:
  void InferProducedDataRegstTimeShape() final;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CUDA_COPY_PEER_TASK_NODE_H_
