#ifndef ONEFLOW_CORE_GRAPH_INSTANCE_STACK_FORWARD_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_INSTANCE_STACK_FORWARD_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/piece_slice_forward_compute_task_node.h"

namespace oneflow {

class InstanceStackForwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InstanceStackForwardCompTaskNode);
  InstanceStackForwardCompTaskNode() = default;
  ~InstanceStackForwardCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kInstanceStackForward; }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  void set_related_piece_slice(PieceSliceForwardCompTaskNode* val) { related_piece_slice_ = val; }

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;

  PieceSliceForwardCompTaskNode* related_piece_slice_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_INSTANCE_STACK_FORWARD_TASK_NODE_H_
