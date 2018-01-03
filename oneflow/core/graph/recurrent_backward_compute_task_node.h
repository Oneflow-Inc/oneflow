#ifndef ONEFLOW_CORE_GRAPH_RECURRENT_BACKWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_RECURRENT_BACKWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/backward_compute_task_node.h"

namespace oneflow {

class RecurrentBackwardCompTaskNode final : public BackwardCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecurrentBackwardCompTaskNode);
  RecurrentBackwardCompTaskNode() = default;
  ~RecurrentBackwardCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kRecurrentBackward; }

 private:
  void VirtualConsumeInRegst();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_RECURRENT_BACKWARD_COMPUTE_TASK_NODE_H_
