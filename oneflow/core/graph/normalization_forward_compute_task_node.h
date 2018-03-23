#ifndef ONEFLOW_CORE_GRAPH_NORMALIZATION_FORWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NORMALIZATION_FORWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/forward_compute_task_node.h"

namespace oneflow {

class NormalizationForwardCompTaskNode final : public ForwardCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationForwardCompTaskNode);
  NormalizationForwardCompTaskNode() = default;
  ~NormalizationForwardCompTaskNode() = default;

  TaskType GetTaskType() const override {
    return TaskType::kNormalizationForward;
  }
  bool IsReadyForBuild() override;

 private:
  void VirtualConsumeRegstOnInEdge(TaskEdge* edge) override;
  void VirtualProduceRegstOnOutEdge(TaskEdge* edge) override;
  void VirtualBuildExecGphStructAndBindInRegst() override;
  void VirtualBuildOutRegst() override;
  void VirtualBuildExtraRegsts() override;
  void VirtualLockExtraRegsts() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NORMALIZATION_FORWARD_COMPUTE_TASK_NODE_H_
