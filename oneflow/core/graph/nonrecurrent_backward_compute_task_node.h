#ifndef ONEFLOW_CORE_GRAPH_NONRECURRENT_BACKWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NONRECURRENT_BACKWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/backward_compute_task_node.h"

namespace oneflow {

class NonRecurrentBackwardCompTaskNode final : public BackwardCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NonRecurrentBackwardCompTaskNode);
  NonRecurrentBackwardCompTaskNode() = default;
  ~NonRecurrentBackwardCompTaskNode() = default;

  TaskType GetTaskType() const override {
    return TaskType::kNonRecurrentBackward;
  }

 private:
  void BuildExecGphAndBindOutDiffRegst() override;
  void BuildInDiffRegst() override;
  void VirtualConsumeInRegst() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NONRECURRENT_BACKWARD_COMPUTE_TASK_NODE_H_
