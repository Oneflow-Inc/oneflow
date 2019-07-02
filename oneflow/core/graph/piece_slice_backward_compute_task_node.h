#ifndef ONEFLOW_CORE_GRAPH_PIECE_SLICE_BACKWARD_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_PIECE_SLICE_BACKWARD_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class PieceSliceBackwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PieceSliceBackwardCompTaskNode);
  PieceSliceBackwardCompTaskNode() = default;
  ~PieceSliceBackwardCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kPieceSliceBackward; }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_PIECE_SLICE_BACKWARD_TASK_NODE_H_
