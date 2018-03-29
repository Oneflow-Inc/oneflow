#ifndef ONEFLOW_CORE_GRAPH_GATHER_FORWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_GATHER_FORWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class GatherBackwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherBackwardCompTaskNode);
  GatherBackwardCompTaskNode() = default;
  ~GatherBackwardCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;

  TaskType GetTaskType() const override { return TaskType::kGatherBackward; }
 private:
};

} // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_GATHER_FORWARD_COMPUTE_TASK_NODE_H_
