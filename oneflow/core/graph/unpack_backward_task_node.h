#ifndef ONEFLOW_CORE_GRAPH_UNPACK_BACKWARD_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_UNPACK_BACKWARD_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class UnpackBackwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnpackBackwardCompTaskNode);
  UnpackBackwardCompTaskNode() = default;
  ~UnpackBackwardCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kUnpackBackward; }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_UNPACK_BACKWARD_TASK_NODE_H_
