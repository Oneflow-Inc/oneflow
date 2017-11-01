#ifndef ONEFLOW_CORE_GRAPH_BACKWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_BACKWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class BwCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BwCompTaskNode);
  BwCompTaskNode() = default;
  ~BwCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  TodoTaskType GetTaskType() const override { return TodoTaskType::kBackward; }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BACKWARD_COMPUTE_TASK_NODE_H_
