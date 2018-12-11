#ifndef ONEFLOW_CORE_GRAPH_TICK_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_TICK_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class TickCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TickCompTaskNode);
  TickCompTaskNode() = default;
  ~TickCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kTick; }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TICK_TASK_NODE_H_
