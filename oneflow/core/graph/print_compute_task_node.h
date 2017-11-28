#ifndef ONEFLOW_CORE_GRAPH_PRINT_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_PRINT_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/sink_compute_task_node.h"

namespace oneflow {

class PrintCompTaskNode final : public SinkCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PrintCompTaskNode);
  PrintCompTaskNode() = default;
  ~PrintCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kPrint; }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_PRINT_COMPUTE_TASK_NODE_H_
