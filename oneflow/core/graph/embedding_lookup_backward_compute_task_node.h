#ifndef ONEFLOW_CORE_GRAPH_EMBEDDING_LOOKUP_BACKWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_EMBEDDING_LOOKUP_BACKWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/backward_compute_task_node.h"

namespace oneflow {

class EmbeddingLookupBackwardCompTaskNode final : public BackwardCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmbeddingLookupBackwardCompTaskNode);
  EmbeddingLookupBackwardCompTaskNode() = default;
  ~EmbeddingLookupBackwardCompTaskNode() = default;

  TaskType GetTaskType() const override {
    return TaskType::kEmbeddingLookupBackward;
  }

 private:
  void VirtualBuildExecGphAndBindOutDiffRegst() override;
  void VirtualBuildActivationDiffRegst() override;
  void VirtualBuildInDiffRegst() override;
  void VirtualConsumeDiffRegst(TaskEdge* edge) override;
  void VirtualConsumeInRegst() override;
  void VirtualProduceInDiffAndBindEdge(TaskEdge* edge) override;
  void VirtualProduceActivationDiff() override;
  void VirtualConsumeActivation(TaskEdge* edge) override;
  void VirtualInferBlobDescInActivationDiff() override;
  void VirtualInferBlobDescsInProducedRegsts() override {
    BuildModelDiffRegst();
  }
  void BuildModelDiffRegst();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_EMBEDDING_LOOKUP_BACKWARD_COMPUTE_TASK_NODE_H_
