#ifndef ONEFLOW_CORE_GRAPH_ACC_TICK_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_ACC_TICK_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/accumulate_compute_task_node.h"

namespace oneflow {

class AccTickCompTaskNode final : public AccumulateCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccTickCompTaskNode);
  AccTickCompTaskNode() = default;
  ~AccTickCompTaskNode() = default;
  TaskType GetTaskType() const override { return TaskType::kAccTick; }
  void BuildExecGphAndRegst() override;

 private:
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_ACC_TICK_COMPUTE_TASK_NODE_H_
