#ifndef ONEFLOW_CORE_GRAPH_KEEP_HEADER_ONLY_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_KEEP_HEADER_ONLY_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class KeepHeaderOnlyCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KeepHeaderOnlyCompTaskNode);
  KeepHeaderOnlyCompTaskNode() = default;
  ~KeepHeaderOnlyCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kKeepHeaderOnly; }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_KEEP_HEADER_ONLY_COMPUTE_TASK_NODE_H_

