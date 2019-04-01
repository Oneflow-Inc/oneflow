#ifndef ONEFLOW_CORE_GRAPH_ACC_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_ACC_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/accumulate_compute_task_node.h"

namespace oneflow {

class AccCompTaskNode final : public AccumulateCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccCompTaskNode);
  AccCompTaskNode() = default;
  ~AccCompTaskNode() = default;
  TaskType GetTaskType() const override { return TaskType::kAcc; }
  void BuildExecGphAndRegst() override;

 private:
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_ACC_COMPUTE_TASK_NODE_H_
