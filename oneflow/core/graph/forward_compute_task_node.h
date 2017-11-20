#ifndef ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ForwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForwardCompTaskNode);
  ForwardCompTaskNode() = default;
  ~ForwardCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  TaskType GetTaskType() const override { return TaskType::kForward; }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_
