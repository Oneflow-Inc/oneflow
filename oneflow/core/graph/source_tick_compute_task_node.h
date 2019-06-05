#ifndef ONEFLOW_CORE_GRAPH_SOURCE_TICK_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_SOURCE_TICK_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class SourceTickCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SourceTickCompTaskNode);
  SourceTickCompTaskNode() = default;
  ~SourceTickCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override {}
  void BuildExecGphAndRegst() override;
  bool IsMeaningLess() override { return false; }

  TaskType GetTaskType() const override { return TaskType::kSourceTick; }

 private:
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_SOURCE_TICK_COMPUTE_TASK_NODE_H_
