#ifndef ONEFLOW_CORE_GRAPH_LOSS_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_LOSS_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class LossCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossCompTaskNode);
  LossCompTaskNode() = default;
  ~LossCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  TodoTaskType GetTaskType() const override { return TodoTaskType::kLoss; }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOSS_COMPUTE_TASK_NODE_H_
