#ifndef ONEFLOW_CORE_GRAPH_INSTANCE_STACK_BACKWARD_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_INSTANCE_STACK_BACKWARD_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class InstanceStackBackwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InstanceStackBackwardCompTaskNode);
  InstanceStackBackwardCompTaskNode() = default;
  ~InstanceStackBackwardCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kInstanceStackBackward; }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_INSTANCE_STACK_BACKWARD_TASK_NODE_H_
