#ifndef ONEFLOW_CORE_GRAPH_BOXING_COPY_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_BOXING_COPY_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class BoxingCopyTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingCopyTaskNode);
  BoxingCopyTaskNode() = default;
  ~BoxingCopyTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kBoxingCopy; }

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;

  std::vector<const TaskNode*> sorted_pred_task_node_vec_;
  std::shared_ptr<Operator> op_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_COPY_TASK_NODE_H_
