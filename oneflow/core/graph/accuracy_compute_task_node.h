#ifndef ONEFLOW_CORE_GRAPH_ACCURAY_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_ACCURAY_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class AccuracyCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccuracyCompTaskNode);
  AccuracyCompTaskNode() = default;
  ~AccuracyCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;
  TaskType GetTaskType() const override { return TaskType::kAccuracy; }

 private:
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_ACCURAY_COMPUTE_TASK_NODE_H_
