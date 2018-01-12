#ifndef ONEFLOW_CORE_GRAPH_NORMAL_BACKWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NORMAL_BACKWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/backward_compute_task_node.h"

namespace oneflow {

class NormalBackwardCompTaskNode final : public BackwardCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalBackwardCompTaskNode);
  NormalBackwardCompTaskNode() = default;
  ~NormalBackwardCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kNormalBackward; }

 private:
  void VirtualBuildExecGphAndBindOutDiffRegst() override;
  void VirtualBuildActivationDiffRegst() override;
  void VirtualBuildInDiffRegst() override;
  void VirtualConsumeInRegst() override;
  void VirtualProduceInDiffAndBindEdge(TaskEdge* edge) override {
    edge->AddRegst("in_diff", ProduceRegst("in_diff"));
  }
  void VirtualProduceActivationDiff() override {
    ProduceRegst("activation_diff", 1, 1);
  }
  void VirtualConsumeActivation(TaskEdge* edge) override {
    ConsumeRegst("activation", edge->GetRegst("activation"));
  }
  void VirtualInferBlobDescInActivationDiff() override {
    auto activation_diff_regst = GetProducedRegst("activation_diff");
    activation_diff_regst->CopyBlobDescWithoutAddLbn(
        GetConsumedRegst("activation").get());
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NORMAL_BACKWARD_COMPUTE_TASK_NODE_H_
