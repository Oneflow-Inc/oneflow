#ifndef ONEFLOW_CORE_GRAPH_NORMAL_BACKWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NORMAL_BACKWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class NormalBackwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalBackwardCompTaskNode);
  NormalBackwardCompTaskNode() = default;
  ~NormalBackwardCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;
  TaskType GetTaskType() const override { return TaskType::kNormalBackward; }
  void RmUselessConsumeRelationshipToFw();

 protected:
  void BuildExecGphAndBindOutDiffRegst();
  void BuildActivationDiffRegst();
  void BuildInDiffRegst();

 private:
  void FixPackedBlobDescOfProducedRegst() override;
  void LinkFwExecNode();
  void BindModelDiffRegst();
  void BindInRegst();
  void InferBlobDescsInProducedRegsts();
  CompTaskNode* GetRelatedFwTaskNode();
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NORMAL_BACKWARD_COMPUTE_TASK_NODE_H_
