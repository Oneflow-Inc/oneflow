#ifndef ONEFLOW_CORE_GRAPH_PIECE_SLICE_FORWARD_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_PIECE_SLICE_FORWARD_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class PieceSliceForwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PieceSliceForwardCompTaskNode);
  PieceSliceForwardCompTaskNode() = default;
  ~PieceSliceForwardCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kPieceSliceForward; }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_PIECE_SLICE_FORWARD_TASK_NODE_H_
