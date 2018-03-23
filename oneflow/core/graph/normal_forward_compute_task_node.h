#ifndef ONEFLOW_CORE_GRAPH_NORMAL_FORWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NORMAL_FORWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/forward_compute_task_node.h"

namespace oneflow {

class NormalForwardCompTaskNode final : public ForwardCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalForwardCompTaskNode);
  NormalForwardCompTaskNode() = default;
  ~NormalForwardCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kNormalForward; }
  bool IsReadyForBuild() override;

 private:
  void VirtualConsumeRegstOnInEdge(TaskEdge* edge) override;
  void VirtualBuildExecGphStructAndBindInRegst() override;
  void VirtualBuildOutRegst() override;
  void VirtualBuildExtraRegsts() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NORMAL_FORWARD_COMPUTE_TASK_NODE_H_
