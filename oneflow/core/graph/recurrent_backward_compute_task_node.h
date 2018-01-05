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
  bool IsReadyForBuild() override;

 private:
  void VirtualBuildExecGphAndBindOutDiffRegst() override;
  void VirtualBuildInDiffRegst() override;
  void VirtualProduceInDiffAndBindEdge(TaskEdge* edge) override;
  void VirtualProduceRegstOnSelfEdge(TaskEdge* edge) override {
    edge->AddRegst("ht_1_diff", ProduceRegst("ht_1_diff"));
  }
  void VirtualConsumeInRegst();
  void VirtualConsumeRegstOnSelfEdge(TaskEdge* edge) override {
    ConsumeRegst("ht_1_diff", edge->GetRegst("ht_1_diff"));
  }
  void VirtualInferBlobDescInHiddenDiff() override;
  bool CanBindInDiffWhenRecurrent(TaskEdge* edge);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_RECURRENT_BACKWARD_COMPUTE_TASK_NODE_H_
