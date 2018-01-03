#ifndef ONEFLOW_CORE_GRAPH_RECURRENT_FORWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_RECURRENT_FORWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/forward_compute_task_node.h"

namespace oneflow {

class RecurrentForwardCompTaskNode final : public ForwardCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecurrentForwardCompTaskNode);
  RecurrentForwardCompTaskNode() = default;
  ~RecurrentForwardCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kRecurrentForward; }
  void VirtualConsumeInRegst(TaskEdge* edge) override;
  bool IsReadyForBuild() override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_RECURRENT_FORWARD_COMPUTE_TASK_NODE_H_
