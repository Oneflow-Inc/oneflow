#ifndef ONEFLOW_CORE_GRAPH_LOSS_PRINT_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_LOSS_PRINT_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/sink_compute_task_node.h"

namespace oneflow {

class LossPrintCompTaskNode final : public SinkCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossPrintCompTaskNode);
  LossPrintCompTaskNode() = default;
  ~LossPrintCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kLossPrint; }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOSS_PRINT_COMPUTE_TASK_NODE_H_
