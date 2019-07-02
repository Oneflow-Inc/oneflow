#ifndef ONEFLOW_CORE_GRAPH_FOREIGN_OUTPUT_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_FOREIGN_OUTPUT_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/foreign_io_compute_task_node.h"

namespace oneflow {

class ForeignOutputCompTaskNode final : public ForeignIOCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForeignOutputCompTaskNode);
  ForeignOutputCompTaskNode() = default;
  ~ForeignOutputCompTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kForeignOutput; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_FOREIGN_OUTPUT_COMPUTE_TASK_NODE_H_
